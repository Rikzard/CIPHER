"""Bounded, in-memory extraction of paragraphs and table cells from DOCX files."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath
from typing import Any, Iterator, Protocol
import unicodedata
from uuid import UUID, uuid4
import zipfile
from xml.etree import ElementTree

from backend.models.contracts import TrustClassification

DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MAX_DOCX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_DOCX_ARCHIVE_ENTRIES = 4096
MAX_EXTRACTED_SEGMENTS = 2000
MAX_EXTRACTED_CHARACTERS = 250_000
MAX_CONTENT_TYPES_XML_BYTES = 1024 * 1024
DOCX_MAIN_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
)


class DocumentIngestionError(ValueError):
    """Base error for rejected or unprocessable document uploads."""


class UnsupportedDocumentTypeError(DocumentIngestionError):
    """Raised when a file is not named as a DOCX document."""


class InvalidDocxError(DocumentIngestionError):
    """Raised when bytes do not form a readable, safe DOCX package."""


class EmptyDocumentError(DocumentIngestionError):
    """Raised when a valid DOCX contains no non-whitespace paragraph/cell text."""


class DocumentContentTooLarge(DocumentIngestionError):
    """Raised when file, expanded archive, or extracted text bounds are exceeded."""


@dataclass(frozen=True, slots=True)
class DocumentContentUnit:
    """Text from one paragraph or one table cell with zero-based source indices."""

    document_id: UUID
    filename: str
    content_type: str
    trust_classification: TrustClassification
    extracted_text: str
    paragraph_index: int | None = None
    table_index: int | None = None
    row_index: int | None = None
    column_index: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Validated DOCX metadata and text segments, retained only in memory."""

    document_id: UUID
    filename: str
    content_type: str
    trust_classification: TrustClassification
    segments: tuple[DocumentContentUnit, ...]


# Keep the earlier internal name available to callers while exposing the
# stable content-unit abstraction independent of python-docx objects.
ExtractedTextSegment = DocumentContentUnit


class _DocumentBody(Protocol):
    def iter_inner_content(self) -> Iterator[Any]: ...


def _safe_filename(filename: str) -> str:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    basename = unicodedata.normalize("NFC", basename)
    basename = "".join(
        "_" if unicodedata.category(char).startswith("C") else char
        for char in basename
    ).strip().strip(".")
    if not basename or len(basename) > 255:
        raise UnsupportedDocumentTypeError("A valid DOCX filename is required")
    if PurePath(basename).suffix.lower() != ".docx":
        raise UnsupportedDocumentTypeError("Only .docx files are supported")
    return basename


def _validate_archive(data: bytes) -> None:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_DOCX_ARCHIVE_ENTRIES:
                raise DocumentContentTooLarge("DOCX package has too many archive entries")
            if sum(item.file_size for item in entries) > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise DocumentContentTooLarge("DOCX package expands beyond the supported size")
            if any(item.flag_bits & 0x1 for item in entries):
                raise InvalidDocxError("Encrypted DOCX packages are not supported")
            names = {item.filename for item in entries}
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise InvalidDocxError("The uploaded file is not a DOCX document")
            if "word/vbaProject.bin" in names:
                raise InvalidDocxError("Macro-enabled documents are not supported")
            content_types_info = archive.getinfo("[Content_Types].xml")
            if content_types_info.file_size > MAX_CONTENT_TYPES_XML_BYTES:
                raise InvalidDocxError("The DOCX content-types manifest is invalid")
            content_types = ElementTree.fromstring(archive.read(content_types_info))
            main_part_types = [
                element.attrib.get("ContentType", "")
                for element in content_types
                if element.attrib.get("PartName") == "/word/document.xml"
            ]
            if DOCX_MAIN_CONTENT_TYPE not in main_part_types:
                raise InvalidDocxError("The package is not a standard DOCX document")
    except (DocumentContentTooLarge, InvalidDocxError):
        raise
    except (
        OSError,
        RuntimeError,
        EOFError,
        NotImplementedError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        ValueError,
        ElementTree.ParseError,
    ) as exc:
        raise InvalidDocxError("The uploaded file is not a readable DOCX document") from exc


def _iter_segments(document: _DocumentBody, document_id: UUID, filename: str) -> Iterator[DocumentContentUnit]:
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    paragraph_index = 0
    table_index = 0
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            yield DocumentContentUnit(
                document_id=document_id,
                filename=filename,
                content_type=DOCX_CONTENT_TYPE,
                trust_classification=TrustClassification.applicant_data(),
                extracted_text=block.text,
                paragraph_index=paragraph_index,
            )
            paragraph_index += 1
        elif isinstance(block, Table):
            for row_index, row in enumerate(block.rows):
                seen_cells: set[int] = set()
                for column_index, cell in enumerate(row.cells):
                    # Merged cells appear at multiple grid positions; emit their
                    # text once and retain the first encountered row/column.
                    cell_identity = id(cell._tc)
                    if cell_identity in seen_cells:
                        continue
                    seen_cells.add(cell_identity)
                    yield DocumentContentUnit(
                        document_id=document_id,
                        filename=filename,
                        content_type=DOCX_CONTENT_TYPE,
                        trust_classification=TrustClassification.applicant_data(),
                        extracted_text=cell.text,
                        table_index=table_index,
                        row_index=row_index,
                        column_index=column_index,
                    )
            table_index += 1


def extract_docx(
    data: bytes,
    *,
    filename: str,
    max_size_bytes: int,
    document_id: UUID | None = None,
) -> ExtractedDocument:
    """Parse a DOCX from memory and return source-located text without saving it."""
    safe_filename = _safe_filename(filename)
    if max_size_bytes <= 0:
        raise ValueError("max_size_bytes must be positive")
    if not data:
        raise EmptyDocumentError("The uploaded file is empty")
    if len(data) > max_size_bytes:
        raise DocumentContentTooLarge("The uploaded file exceeds the configured size limit")

    _validate_archive(data)
    try:
        from docx import Document

        parsed = Document(BytesIO(data))
    except Exception as exc:
        raise InvalidDocxError("The uploaded file is not a readable DOCX document") from exc

    resolved_document_id = document_id or uuid4()
    segments: list[DocumentContentUnit] = []
    total_characters = 0
    for segment in _iter_segments(parsed, resolved_document_id, safe_filename):
        total_characters += len(segment.extracted_text)
        if total_characters > MAX_EXTRACTED_CHARACTERS:
            raise DocumentContentTooLarge("Extracted document text exceeds the supported limit")
        segments.append(segment)
        if len(segments) > MAX_EXTRACTED_SEGMENTS:
            raise DocumentContentTooLarge("DOCX contains too many text segments")

    if not any(segment.extracted_text.strip() for segment in segments):
        raise EmptyDocumentError("The DOCX contains no extractable text")

    return ExtractedDocument(
        document_id=resolved_document_id,
        filename=safe_filename,
        content_type=DOCX_CONTENT_TYPE,
        trust_classification=TrustClassification.applicant_data(),
        segments=tuple(segments),
    )
