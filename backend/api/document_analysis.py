"""DOCX upload endpoint for source-located CIPHER analysis."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from backend.api.analysis import get_application_service
from backend.application.orchestrator import ApplicationService
from backend.config import Settings
from backend.document.docx_parser import (
    DocumentContentTooLarge,
    EmptyDocumentError,
    InvalidDocxError,
    UnsupportedDocumentTypeError,
)
from backend.document.service import DocumentAnalysisResponse, DocumentAnalysisService

router = APIRouter()


@router.post("/analyze-document", response_model=DocumentAnalysisResponse, tags=["document analysis"])
async def analyze_document(
    request: Request,
    file: UploadFile = File(..., description="Candidate-submitted DOCX file"),
    application_service: ApplicationService = Depends(get_application_service),
) -> DocumentAnalysisResponse:
    """Extract local DOCX text and run it through the shared CIPHER pipeline."""
    filename = file.filename or ""
    if not filename.lower().endswith(".docx"):
        await file.close()
        raise HTTPException(status_code=415, detail="Only .docx files are supported")

    settings: Settings = request.app.state.settings
    started_at = time.perf_counter()
    try:
        data = await file.read(settings.max_document_size_bytes + 1)
        if not data:
            raise HTTPException(status_code=422, detail="The uploaded file is empty")
        if len(data) > settings.max_document_size_bytes:
            raise HTTPException(status_code=413, detail="The uploaded file exceeds the configured size limit")

        response = DocumentAnalysisService(application_service).analyze_docx(
            data,
            filename=filename,
            max_size_bytes=settings.max_document_size_bytes,
            processing_latency_ms=0.0,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return response.model_copy(update={"processing_latency_ms": latency_ms})
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from None
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except DocumentContentTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from None
    except InvalidDocxError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    finally:
        await file.close()
