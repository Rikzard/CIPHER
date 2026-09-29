# CIPHER Architecture

## Purpose and scope

CIPHER is a gateway for assessing prompt injection risk in content before an integrating LLM application uses that content. The backend uses Python with FastAPI. The implemented direct-text analysis path normalizes input, runs `RuleDetector` and `EmbeddingDetector`, fuses available scores, and applies the existing decision policy. The embedding detector uses a locally configured Sentence Transformer and FAISS index.

The gateway returns a decision and bounded evidence. It is not an authorization system, a content sanitizer, or a guarantee that a prompt is safe. The calling application must enforce the decision and independently constrain model and tool permissions.

## High-level request flow

```text
LLM application
    -> FastAPI adapter / request validation
    -> application service
        -> input normalization
        -> detector ports (RuleDetector + EmbeddingDetector)
        -> risk score fusion
        -> decision engine
    <- assessment response
    -> structured prompt/data builder (when preparing model context)
    -> LLM application / model

Offline evaluation -> detector and policy contracts
Future indirect injection scanner -> retrieved documents / tool outputs
```

The API adapter should not contain detection logic. The application service coordinates detector execution through `BaseDetector`; the application composition boundary supplies the rule and embedding detectors. Evaluation and future indirect scanning can reuse detector contracts without becoming hidden side effects of an API request.

## Module boundaries

| Module | Responsibility | Inputs and outputs | Boundary rules |
|---|---|---|---|
| API adapter (`backend/api`) | FastAPI routes, request/response validation, request IDs, HTTP error mapping | Typed API request to application service; typed response to caller | No detector algorithms, threshold policy, prompt logging, or model calls in routes |
| Application service (`backend/application`) | Orchestrate normalization, detector execution, fusion, and policy evaluation | Request context to normalized content, detector evidence, fused risk, and decision | Executes each configured `BaseDetector` independently on the same normalized content; detector algorithms remain in detector modules |
| Input normalization (`backend/normalization`) | Produce a canonical text representation and source metadata for analysis | Raw text plus source type/locale hints to normalized content | Preserve exact original text separately; normalization must not silently rewrite downstream content |
| Rule-based detection (`backend/detector/rules`) | Deterministic indicators such as configured banned terms and suspicious structural patterns | Normalized content to findings with rule IDs, spans where safe, and confidence/severity | Version rules; avoid treating a keyword match alone as proof of malicious intent |
| Embedding similarity (`backend/detector/embeddings`) | Compare content with a curated reference set or policy examples | Normalized content to similarity scores and reference IDs | Local model and FAISS index are provisioned and validated offline before serving; load once at app construction and never download/rebuild per request; do not expose raw reference text in API output |
| ML classifier (`backend/detector/classifier`) | Later, classify injection likelihood using a trained model | Normalized content to calibrated probability/class label and model version | Deferred; no model, weights, or dependency is part of this documentation baseline |
| Risk score fusion (`backend/risk`) | Combine detector outputs into a stable risk assessment | Versioned detector evidence and calibration to risk score/band | Keep evidence and score; handle missing/unavailable signals explicitly; avoid accidental double counting |
| Decision engine (`backend/policy`) | Apply policy thresholds and optional source-specific rules | Risk assessment plus policy version/context to `allow`, `flag`, or `block` | Separate configurable policy from detector scoring; decisions must be deterministic for identical inputs/configuration |
| Structured prompt/data separation (`backend/prompting`) | Represent trusted instructions and untrusted content as distinct typed sections for downstream prompt construction | Trusted template plus untrusted payload to a structured envelope | Never promote analyzed text into trusted instructions; encoding/serialization must preserve the boundary |
| Evaluation (`evaluation`) | Maintain fixtures, adversarial scenarios, metrics, and reproducible offline runs | Labeled cases and versioned detector/policy config to reports | Keep evaluation out of production request paths; measure false positives and false negatives as well as aggregate scores |
| Indirect injection scanner (`backend/scanning`, future) | Inspect retrieved documents, tool results, and other external content before agent use | Content plus provenance/source metadata to detector evidence | Future boundary; scan each external source independently and preserve provenance; not implemented in initial scope |

The implemented backend follows these module boundaries. ML classification and indirect scanning remain deferred.

## Core contracts (conceptual)

The implementation should define typed contracts approximately equivalent to:

- `AnalysisInput`: original text reference/content, source kind (`user`, `retrieval`, `tool`, or other configured source), and request metadata.
- `NormalizedContent`: canonical analysis text, normalization version, and mappings back to source offsets when feasible.
- `DetectorResult`: detector name/version, availability, score or findings, and bounded rationale metadata.
- `RiskAssessment`: fused score/band, contributing detector versions, and calibration/fusion version.
- `Decision`: `allow`, `flag`, or `block`, policy version, and stable reason codes.
- `PromptEnvelope`: trusted instruction fields and explicitly marked untrusted-data fields, serialized through a well-defined adapter.

Exact field names and score ranges are implementation decisions. Contracts must distinguish an unavailable detector from a detector that evaluated content and found no signal.

## Scoring and decisions

Detector outputs are evidence, not decisions. The current `RiskFusion` baseline takes the maximum score from available detectors; unavailable and score-less results do not contribute a score but remain in the detector result list and contributing-version metadata. If no detector has an available score, this implementation returns `0.0`/`LOW`; that existing fallback can lead the decision engine to `allow`, so callers must inspect detector availability. `EmbeddingDetector` reports cosine similarity as a 0–100 signal using an initial configurable threshold of `0.65`; the threshold is explicitly uncalibrated and requires evaluation. No risk or decision thresholds were changed for the hybrid integration.

Response reasons should use stable codes and concise metadata. Avoid returning hidden system prompts, full matched reference examples, or unnecessary spans of user content. Any score shown to callers should be documented as a calibrated operational signal only when calibration has been measured.

## Structured instruction and data boundary

CIPHER should model trusted instructions and untrusted text as separate fields before any downstream serialization. An integration adapter is responsible for mapping those fields to the target model's supported structured message format. Delimiters or labels can improve clarity but are not a security boundary on their own. The downstream application must retain tool authorization checks and should not treat text from a user, retrieval result, or tool as an instruction source.

## API boundary

The HTTP surface exposes `GET /health`, which returns process status plus service, version, and environment metadata. It does not claim detector readiness. `POST /analyze` runs the current hybrid `RuleDetector` + `EmbeddingDetector` path and returns both detector results, fused risk, decision, findings/categories, and latency. An unavailable embedding model or index is represented with `available: false` and `score: null`; the API does not convert that result into a successful zero score. Authentication, rate limiting, tenancy, and persistence remain outside the current implementation. Raw prompts should not be logged by default. Unexpected server errors should return generic details and must not expose exception text or prompt data.

## Extensibility and dependencies

Use Python type contracts and dependency injection at module boundaries. FastAPI belongs to the API adapter. Sentence Transformers and FAISS are isolated within the embedding detector; the model and index paths are configurable and loaded outside request handling. The offline-capable provisioning/validation workflow, dataset manifest, metadata schema, and developer setup commands are described in [EMBEDDINGS.md](EMBEDDINGS.md). No model weights or generated indexes are bundled. ML classification and external embedding services remain deferred.

## Research mapping

- HOUYI motivates testing context, separator, and disruptor variations as adversarial inputs rather than relying on one string pattern.
- Detection work on LLM-integrated applications motivates separate banned-term/rule, embedding, classifier, and decision components.
- Work on attacks in LLM and agent systems motivates treating indirect content and tool/retrieval paths as first-class future sources.
- StruQ motivates explicit structure between trusted instructions and untrusted data.
- AgentDojo motivates adversarial, task-oriented evaluation in addition to isolated examples.

These are architectural inspirations. No benchmark result or defense guarantee is asserted here.
