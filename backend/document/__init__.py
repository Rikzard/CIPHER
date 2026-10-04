"""Local DOCX extraction and source provenance for CIPHER."""

from backend.document.docx_parser import (
    DOCX_CONTENT_TYPE,
    DocumentContentUnit,
    DocumentContentTooLarge,
    DocumentIngestionError,
    EmptyDocumentError,
    ExtractedDocument,
    ExtractedTextSegment,
    InvalidDocxError,
    UnsupportedDocumentTypeError,
    extract_docx,
)
from backend.document.service import (
    DocumentAnalysisResponse,
    DocumentFinding,
    DocumentSourceLocation,
)

__all__ = [
    "DOCX_CONTENT_TYPE",
    "DocumentContentUnit",
    "DocumentContentTooLarge",
    "DocumentIngestionError",
    "DocumentAnalysisResponse",
    "DocumentFinding",
    "DocumentSourceLocation",
    "EmptyDocumentError",
    "ExtractedDocument",
    "ExtractedTextSegment",
    "InvalidDocxError",
    "UnsupportedDocumentTypeError",
    "extract_docx",
]
