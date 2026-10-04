# CIPHER DOCX Security Ingestion

## Scope and trust boundary

`POST /analyze-document` accepts one local `.docx` upload and extracts body
paragraphs and top-level table cells. Candidate document content is untrusted
data; HR application instructions remain trusted and separate. The server
assigns trust from the application interface, never from text, filename,
document metadata, XML fields, labels, or natural-language claims. Each
applicant unit is always `APPLICANT`/`DATA`/`UNTRUSTED`. Extracted candidate
text is only passed to CIPHER as data for security analysis. The
upload path does not add HR scoring, candidate-quality evaluation,
LLM calls, URL fetching, document-embedded code execution, or permanent file
storage. Extracted text is passed to CIPHER's existing analysis service; the
document layer does not implement normalization, detectors, score fusion, or
decision thresholds.

The endpoint is a security signal for a calling application. It does not
guarantee that a document is safe, and it does not turn extracted candidate
content into trusted instructions.

## Parsing and resource limits

The parser uses `python-docx` against an in-memory byte stream. It sanitizes the
client filename to a basename, then verifies the ZIP package, required OOXML
parts, and declared Word document content type. A `.docx` suffix alone is not
accepted as proof of format. Encrypted and macro-enabled packages are rejected.
Upload bytes are read only up to the configured maximum plus one byte before
DOCX parsing; the uploaded document is not saved to an application-managed
permanent location.

The default upload limit is 10 MiB. Set `CIPHER_MAX_DOCUMENT_SIZE_BYTES` to a
positive integer to change it. Additional parser bounds limit expanded DOCX
package contents to 100 MiB, archive entries to 4,096, extracted segments to
2,000, and total extracted text to 250,000 characters. The extracted-text
bound matches the existing normalizer's input limit.

The runtime dependencies are `python-docx==1.2.0` for parsing and
`python-multipart==0.0.32`, which FastAPI requires to parse multipart file
uploads. No other document-processing dependencies were added.

## Extracted segments and source locations

Each body paragraph and top-level table cell is represented by a stable
`DocumentContentUnit`, independent of python-docx objects. Every unit carries
the server-assigned trust classification, document ID, sanitized filename,
content type, exact extracted text, and source indices. Non-empty units are
analyzed independently. Paragraph, table, row, and column indices are
zero-based. Empty units are retained by extraction but skipped for analysis. A
document with no non-whitespace text is rejected.

Every segment retains:

- document UUID
- basename-only filename
- verified DOCX media type
- exact extracted text
- paragraph index, or table index plus row and column indices

For each unit, `ApplicationService.analyze_applicant_document_content()`
constructs applicant/untrusted input and runs configured detectors through the
existing normalized-content contract. Normalized content and classifier chunks
inherit that immutable classification; detectors do not set or modify it. Long
paragraphs or cells use the classifier's existing overlapping-window
chunking; no new chunking algorithm is introduced here. The classifier reports
its highest-scoring chunk and source span in detector metadata; the enclosing
`source_location` identifies the originating paragraph or cell.

The existing `fuse_risk()` function evaluates detector evidence separately
for each unit. Document risk is the maximum of those unit risk scores;
unavailable detector scores are ignored by existing fusion. The existing
`evaluate_decision()` function produces the verdict from the maximum risk and
available detector findings. No new document scoring formula or threshold is
introduced.

## Endpoint contract

`POST /analyze-document` accepts `multipart/form-data` with one field named
`file`. Supply a `.docx` filename and a readable DOCX package. Successful
requests return:

- `document_id`, `filename`, `content_type`, and top-level
  `trust_classification` (`APPLICANT`/`DATA`/`UNTRUSTED`)
- `verdict`, `risk_score`, and `risk_band`
- `detector_results`, each wrapping a normal `DetectorResult` with its source
  location
- `findings`, each containing detector name, finding text, and source location
- `attack_categories` from the existing rule and semantic finding formats; the
  binary classifier does not predict categories
- `processing_latency_ms`

Example finding:

```json
{
  "detector_name": "rule_detector",
  "finding": "[RULE-001] Direct Override matched",
  "source_location": {
    "document_id": "d9aa22d7-7f72-4775-a4a1-0d329a07e0a1",
    "filename": "candidate.docx",
    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "paragraph_index": 4,
    "table_index": null,
    "row_index": null,
    "column_index": null
  }
}
```

Invalid extensions or filenames return HTTP 415. A renamed non-DOCX file with
a `.docx` extension is rejected by package/content-type validation. Empty
files and invalid or textless DOCX packages return HTTP 422. Uploads over the
configured byte limit and documents exceeding parser bounds return HTTP 413.
Detector unavailability is represented with `available: false` and
`score: null`; it does not stop other detectors or units from processing.
Extra form fields claiming `source=HR` or trusted status do not affect the
server-assigned applicant classification. The direct-text `/analyze` endpoint
also rejects client-provided source/trust fields and keeps its existing
response shape. `ApplicationService.analyze_hr_instructions()` provides an
internal HR-origin interface; there is no HR HTTP endpoint or authentication
provider yet. A future authenticated adapter must select the HR interface only
after verifying the source/interface.

## Limitations

- Detection is performed per paragraph or table cell. An attack deliberately
  split across separate segments may not be recognized as a single sequence.
- Source locations identify the paragraph or cell. Classifier chunk offsets
  in detector metadata are relative to that unit, not document-wide offsets.
- Headers, footers, nested tables, comments, tracked revisions, text boxes,
  footnotes, embedded images, and scanned-page OCR are not extracted.
- A multipart request may be temporarily spooled by the ASGI framework while
  parsing; CIPHER does not retain the document after the request completes.
- DOCX parsing and detection do not execute macros, embedded scripts, or other
  document code.
- Trust labeling is assigned server-side, but identity authentication and
  authorization are not implemented. The trust label and delimiters alone do
  not make a downstream LLM safe; future integration must preserve distinct
  trusted-instruction and untrusted-data fields.
- The current rule, semantic, and classifier signals can miss attacks and can
  flag benign content. The classifier probabilities and embedding threshold
  remain uncalibrated for production.
