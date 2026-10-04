# CIPHER Architecture

## Purpose and scope

CIPHER is a gateway for assessing prompt injection risk in content before an integrating LLM application uses that content. The backend uses Python with FastAPI. The implemented direct-text analysis path normalizes input, runs `RuleDetector`, `EmbeddingDetector`, and `MLClassifierDetector`, fuses available scores, and applies the existing decision policy. The embedding detector uses a locally configured Sentence Transformer and FAISS index; the classifier loads its trained artifact from a local path.

The document path keeps trusted HR instructions separate from untrusted candidate-submitted DOCX content, extracts body paragraphs and table cells, and sends each text unit through the same application analysis service while retaining source locations and an immutable source/role/trust classification.

The gateway returns a decision and bounded evidence. It is not an authorization system, a content sanitizer, or a guarantee that a prompt is safe. The calling application must enforce the decision and independently constrain model and tool permissions.

## High-level request flow

```text
LLM application
    -> FastAPI adapter / request validation
    -> application service
        -> input normalization
        -> detector ports (RuleDetector + EmbeddingDetector + MLClassifierDetector)
        -> risk score fusion
        -> decision engine
    <- assessment response
    -> structured prompt/data builder (when preparing model context)
    -> LLM application / model

Offline evaluation -> detector and policy contracts
Future indirect injection scanner -> retrieved documents / tool outputs

POST /analyze-document -> in-memory DOCX validation/extraction -> located text segments
    -> ApplicationService per segment -> existing RiskFusion -> DecisionEngine
    -> findings with paragraph/table-cell locations
```

The API adapter should not contain detection logic. The application service coordinates detector execution through `BaseDetector`; the application composition boundary supplies all three detectors. Evaluation and future indirect scanning can reuse detector contracts without becoming hidden side effects of an API request.

## Module boundaries

| Module | Responsibility | Inputs and outputs | Boundary rules |
|---|---|---|---|
| API adapter (`backend/api`) | FastAPI routes, request/response validation, request IDs, HTTP error mapping | Typed API request to application service; typed response to caller | No detector algorithms, threshold policy, prompt logging, or model calls in routes |
| Application service (`backend/application`) | Orchestrate normalization, detector execution, fusion, and policy evaluation | Request context to normalized content, detector evidence, fused risk, and decision | Executes each configured `BaseDetector` independently on the same normalized content; detector algorithms remain in detector modules |
| Input normalization (`backend/normalization`) | Produce a canonical text representation and source metadata for analysis | Raw text plus source type/locale hints to normalized content | Preserve exact original text separately; normalization must not silently rewrite downstream content |
| Trust classification (`backend/models/contracts.py`) | Assign immutable source, content role, and trust level before analysis | Server-side source/interface context to `TrustClassification` | HR interface creates trusted instructions; applicant, user, retrieval, and tool content are untrusted data; semantic text and client-supplied labels cannot promote trust |
| DOCX ingestion (`backend/document`) | Validate and extract untrusted local DOCX content into stable `DocumentContentUnit` values with source metadata and trust classification | DOCX bytes to paragraph/table-cell text and document/paragraph/table/row/column locations | Server assigns `APPLICANT`/`DATA`/`UNTRUSTED`; verify OOXML content type and package; parse in memory; enforce upload/archive/segment/text bounds; never interpret embedded code; no HR logic or LLM calls; analyze each unit through the existing application service and aggregate with maximum unit risk |
| Rule-based detection (`backend/detector/rules`) | Deterministic indicators such as configured banned terms and suspicious structural patterns | Normalized content to findings with rule IDs, spans where safe, and confidence/severity | Version rules; avoid treating a keyword match alone as proof of malicious intent |
| Embedding similarity (`backend/detector/embeddings`) | Compare content with a curated reference set or policy examples | Normalized content to similarity scores and reference IDs | Local model and FAISS index are provisioned and validated offline before serving; load once at app construction and never download/rebuild per request; do not expose raw reference text in API output |
| ML classifier (`backend/detector/classifier`) | Independently classify injection likelihood with a local binary DistilBERT model | `NormalizedContent` to probability-derived score, findings, chunk evidence, and artifact version | Selected by `ApplicationService` and run by `/analyze`; optional local runtime dependencies and the ignored trained artifact are required; unavailable artifacts return `available: false` with no score; probabilities are uncalibrated |
| Risk score fusion (`backend/risk`) | Combine detector outputs into a stable risk assessment | Versioned detector evidence and calibration to risk score/band | Keep evidence and score; handle missing/unavailable signals explicitly; avoid accidental double counting |
| Decision engine (`backend/policy`) | Apply policy thresholds and optional source-specific rules | Risk assessment plus policy version/context to `allow`, `flag`, or `block` | Separate configurable policy from detector scoring; decisions must be deterministic for identical inputs/configuration |
| Structured prompt/data separation (`backend/prompting`) | Represent trusted instructions and untrusted content as distinct typed sections for downstream prompt construction | Trusted template plus untrusted payload to a structured envelope | Never promote analyzed text into trusted instructions; encoding/serialization must preserve the boundary |
| Evaluation (`evaluation`) | Maintain fixtures, adversarial scenarios, metrics, and reproducible offline runs | Labeled cases and versioned detector/policy config to reports | Keep evaluation out of production request paths; measure false positives and false negatives as well as aggregate scores |
| Indirect injection scanner (`backend/scanning`, future) | Inspect retrieved documents, tool results, and other external content before agent use | Content plus provenance/source metadata to detector evidence | Future boundary; scan each external source independently and preserve provenance; not implemented in initial scope |

The implemented backend follows these module boundaries. The ML classifier has an isolated training/evaluation workflow and participates in direct-text and DOCX-segment API analysis. `ApplicationService.analyze_hr_instructions()` is the server-side HR-origin interface; `analyze_applicant_document_content()` always constructs applicant/untrusted trust metadata. No authentication provider exists yet, so a future authenticated adapter must select the appropriate interface from verified identity/context. Indirect scanning remains deferred.

## Core contracts (conceptual)

The implementation should define typed contracts approximately equivalent to:

- `TrustClassification`: immutable server-assigned source (`HR`, `APPLICANT`, `USER`, `RETRIEVAL`, or `TOOL`), content role, and trust level; HR maps only to trusted instructions, other sources map to untrusted data.
- `AnalysisInput`: original text reference/content, source kind, trust classification, and request metadata.
- `NormalizedContent`: canonical analysis text, normalization version, source trust classification, and mappings back to source offsets when feasible.
- `DetectorResult`: detector name/version, availability, score or findings, and bounded rationale metadata.
- `RiskAssessment`: fused score/band, contributing detector versions, and calibration/fusion version.
- `Decision`: `allow`, `flag`, or `block`, policy version, and stable reason codes.
- `PromptEnvelope`: trusted instruction fields and explicitly marked untrusted-data fields, serialized through a well-defined adapter.

Exact field names and score ranges are implementation decisions. Contracts must distinguish an unavailable detector from a detector that evaluated content and found no signal.

## Scoring and decisions

Detector outputs are evidence, not decisions. The existing `RiskFusion` baseline receives all three detector results and takes the maximum score among available results with non-null scores; unavailable and score-less results do not contribute a score but remain in the detector result list and contributing-version metadata. If no detector has an available score, this implementation returns `0.0`/`LOW`; that existing fallback can lead the decision engine to `allow`, so callers must inspect detector availability. `EmbeddingDetector` reports cosine similarity as a 0–100 signal using an initial configurable threshold of `0.65`; the classifier emits an uncalibrated probability-derived 0–100 score using its existing 0.5 finding threshold. Both require further evaluation. No fusion, risk, or decision thresholds were changed for classifier integration.

Response reasons should use stable codes and concise metadata. Avoid returning hidden system prompts, full matched reference examples, or unnecessary spans of user content. Any score shown to callers should be documented as a calibrated operational signal only when calibration has been measured.

## Structured instruction and data boundary

CIPHER models trusted instructions and untrusted text separately before any downstream serialization. `TrustClassification` is immutable and assigned from a server-side interface/context before normalization and detection. `ApplicationService.analyze_hr_instructions()` marks trusted HR instructions; `analyze_applicant_document_content()` hard-codes applicant/untrusted data. The direct-text route assigns user/untrusted data and disallows client source/trust fields. The DOCX route assigns applicant/untrusted data regardless of the document content or multipart fields. Detectors receive the classification as part of `NormalizedContent`, but do not determine or mutate it; classifier chunks inherit the same context, and findings/document assessments retain it with source mapping. An integration adapter must map trusted instructions and untrusted data to distinct target-model fields. Delimiters or labels can improve clarity but are not a security boundary on their own. The downstream application must retain tool authorization checks and should not treat text from a user, retrieval result, or tool as an instruction source.

## API boundary

The HTTP surface exposes `GET /health`, which returns process status plus service, version, and environment metadata. It does not claim detector readiness. `POST /analyze` keeps its text-only contract and assigns `USER`/`DATA`/`UNTRUSTED` server-side; client fields such as `source` or `trust_classification` are rejected. `POST /analyze-document` accepts a bounded multipart DOCX upload and assigns `APPLICANT`/`DATA`/`UNTRUSTED` regardless of document text, filename, metadata, or extra form fields. It extracts paragraphs and table cells in memory as `DocumentContentUnit` values, analyzes each unit through the same app-scoped service, and returns detector evidence and findings with source locations and trust classification. Classifier chunks inherit trust from `NormalizedContent` and remain mapped to their originating unit through source location plus unit-relative chunk spans in detector metadata. Document risk is the maximum existing fused risk across units, ignoring unavailable scores; the existing policy engine produces the verdict. Detectors receive trust metadata but do not assign or mutate it. Missing or invalid embedding resources or classifier artifacts are represented with `available: false` and `score: null`; other detectors and units continue. Authentication, rate limiting, tenancy, and persistence remain outside the current implementation. Raw prompts and uploaded documents should not be logged by default. Unexpected server errors should return generic details and must not expose exception text or prompt data.

## Extensibility and dependencies

Use Python type contracts and dependency injection at module boundaries. FastAPI belongs to the API adapter. `python-docx` and FastAPI's multipart parser support local in-memory document ingestion. Sentence Transformers and FAISS are isolated within the embedding detector; the model and index paths are configurable and loaded outside request handling. The offline-capable provisioning/validation workflow, dataset manifest, metadata schema, and developer setup commands are described in [EMBEDDINGS.md](EMBEDDINGS.md). The classifier uses optional `classifier` and `classifier-train` dependency sets, local-only model loading, a versioned dataset, and a separate workflow described in [ML_CLASSIFIER.md](ML_CLASSIFIER.md). Model files and generated indexes are local artifacts and are not bundled in Git. The classifier is constructed once with the other application-scoped detectors, never loaded from the request route, and has no final decision authority. DOCX request and extraction details are in [DOCUMENT_INGESTION.md](DOCUMENT_INGESTION.md).

## Research mapping

- HOUYI motivates testing context, separator, and disruptor variations as adversarial inputs rather than relying on one string pattern.
- Detection work on LLM-integrated applications motivates separate banned-term/rule, embedding, classifier, and decision components.
- Work on attacks in LLM and agent systems motivates treating indirect content and tool/retrieval paths as first-class future sources.
- StruQ motivates explicit structure between trusted instructions and untrusted data.
- AgentDojo motivates adversarial, task-oriented evaluation in addition to isolated examples.

These are architectural inspirations. No benchmark result or defense guarantee is asserted here.
