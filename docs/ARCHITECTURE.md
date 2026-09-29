# CIPHER Architecture

## Purpose and scope

CIPHER is a gateway design for assessing prompt injection risk in content before an integrating LLM application uses that content. The backend uses Python with FastAPI. The current implementation includes an application factory, environment-backed configuration, and a health route; analysis and detector components remain planned.

The gateway returns a decision and bounded evidence. It is not an authorization system, a content sanitizer, or a guarantee that a prompt is safe. The calling application must enforce the decision and independently constrain model and tool permissions.

## High-level request flow

```text
LLM application
    -> FastAPI adapter / request validation
    -> application service
        -> input normalization
        -> detector ports (rules | embeddings | ML, independently optional)
        -> risk score fusion
        -> decision engine
    <- assessment response
    -> structured prompt/data builder (when preparing model context)
    -> LLM application / model

Offline evaluation -> detector and policy contracts
Future indirect injection scanner -> retrieved documents / tool outputs
```

The API adapter should not contain detection logic. The application service coordinates components but does not implement their algorithms. Evaluation and future indirect scanning can reuse detector contracts without becoming hidden side effects of an API request.

## Module boundaries

| Module | Responsibility | Inputs and outputs | Boundary rules |
|---|---|---|---|
| API adapter (`backend/api`) | FastAPI routes, request/response validation, request IDs, HTTP error mapping | Typed API request to application service; typed response to caller | No detector algorithms, threshold policy, prompt logging, or model calls in routes |
| Application service (`backend/application`) | Orchestrate analysis and prompt preparation use cases | Request context to normalized content, detector evidence, fused risk, and decision | Depends on interfaces, not concrete model libraries |
| Input normalization (`backend/normalization`) | Produce a canonical text representation and source metadata for analysis | Raw text plus source type/locale hints to normalized content | Preserve exact original text separately; normalization must not silently rewrite downstream content |
| Rule-based detection (`backend/detectors/rules`) | Deterministic indicators such as configured banned terms and suspicious structural patterns | Normalized content to findings with rule IDs, spans where safe, and confidence/severity | Version rules; avoid treating a keyword match alone as proof of malicious intent |
| Embedding similarity (`backend/detectors/embeddings`) | Compare content with a curated reference set or policy examples | Normalized content to similarity scores and reference IDs | Embedding backend and reference set are replaceable; do not expose raw reference text in API output |
| ML classifier (`backend/detectors/classifier`) | Later, classify injection likelihood using a trained model | Normalized content to calibrated probability/class label and model version | Deferred; no model, weights, or dependency is part of this documentation baseline |
| Risk score fusion (`backend/risk`) | Combine detector outputs into a stable risk assessment | Versioned detector evidence and calibration to risk score/band | Keep evidence and score; handle missing/unavailable signals explicitly; avoid accidental double counting |
| Decision engine (`backend/policy`) | Apply policy thresholds and optional source-specific rules | Risk assessment plus policy version/context to `allow`, `flag`, or `block` | Separate configurable policy from detector scoring; decisions must be deterministic for identical inputs/configuration |
| Structured prompt/data separation (`backend/prompting`) | Represent trusted instructions and untrusted content as distinct typed sections for downstream prompt construction | Trusted template plus untrusted payload to a structured envelope | Never promote analyzed text into trusted instructions; encoding/serialization must preserve the boundary |
| Evaluation (`evaluation`) | Maintain fixtures, adversarial scenarios, metrics, and reproducible offline runs | Labeled cases and versioned detector/policy config to reports | Keep evaluation out of production request paths; measure false positives and false negatives as well as aggregate scores |
| Indirect injection scanner (`backend/scanning`, future) | Inspect retrieved documents, tool results, and other external content before agent use | Content plus provenance/source metadata to detector evidence | Future boundary; scan each external source independently and preserve provenance; not implemented in initial scope |

Directory names are proposed and may be refined during implementation without collapsing the responsibilities above.

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

Detector outputs are evidence, not decisions. The fusion layer owns score normalization and calibration. The decision engine owns thresholds, source-sensitive policy, and the mapping to `allow`, `flag`, or `block`. Both should be versioned so evaluations can reproduce a historical decision. A missing or failed detector should follow explicit policy; it must not silently count as a clean result.

Response reasons should use stable codes and concise metadata. Avoid returning hidden system prompts, full matched reference examples, or unnecessary spans of user content. Any score shown to callers should be documented as a calibrated operational signal only when calibration has been measured.

## Structured instruction and data boundary

CIPHER should model trusted instructions and untrusted text as separate fields before any downstream serialization. An integration adapter is responsible for mapping those fields to the target model's supported structured message format. Delimiters or labels can improve clarity but are not a security boundary on their own. The downstream application must retain tool authorization checks and should not treat text from a user, retrieval result, or tool as an instruction source.

## API boundary

The current HTTP surface exposes `GET /health`, which returns process status plus service, version, and environment metadata. It does not claim detector readiness. Future API work should add a narrow synchronous analysis operation whose response includes a request ID, decision, risk assessment, and bounded reason codes. Authentication, rate limiting, tenancy, persistence, and deployment configuration are integration concerns to specify before exposing a service outside a trusted environment. Raw prompts should not be logged by default. Unexpected server errors should return generic details and must not expose exception text or prompt data.

## Extensibility and dependencies

Use Python type contracts and dependency injection at module boundaries. FastAPI belongs to the API adapter. Detector implementations may later need specialized packages, but those packages must not leak into the API, policy, or evaluation contracts. Prefer a local deterministic rules implementation as the first detector phase; defer external embedding services and model runtimes until their data handling and operational requirements are chosen.

## Research mapping

- HOUYI motivates testing context, separator, and disruptor variations as adversarial inputs rather than relying on one string pattern.
- Detection work on LLM-integrated applications motivates separate banned-term/rule, embedding, classifier, and decision components.
- Work on attacks in LLM and agent systems motivates treating indirect content and tool/retrieval paths as first-class future sources.
- StruQ motivates explicit structure between trusted instructions and untrusted data.
- AgentDojo motivates adversarial, task-oriented evaluation in addition to isolated examples.

These are architectural inspirations. No benchmark result or defense guarantee is asserted here.
