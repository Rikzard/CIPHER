# CIPHER Specification

## Scope and implementation status

This document describes the repository as it is currently implemented. The backend is a minimal Python/FastAPI application with environment-backed settings and a health endpoint. It does **not** currently accept content for prompt injection analysis. Detection, scoring, policy decisions, and prompt construction described in the architecture are not implemented.

## Purpose

CIPHER is intended to become a defense-in-depth gateway that helps an integrating LLM application assess untrusted text for prompt injection risk and make an explicit policy decision. The current implementation only confirms that the API process responds and reports configured service metadata. It provides no security verdict and does not protect an LLM application by itself.

## Threat model

The design treats user text and, in a future agent integration, retrieved documents and tool outputs as untrusted. The documented adversary may provide arbitrary text, obfuscated or paraphrased instructions, quoted content, or content that attempts to override higher-priority instructions or cause disclosure or unauthorized actions.

These are design threats, not threats currently detected by the running API. CIPHER currently has no analysis endpoint, detector, tool integration, model integration, authentication, authorization, rate limiting, or prompt storage. The calling application remains responsible for access control, tool permissions, secrets, and enforcement. An `allow` result or safety guarantee is not produced by the current implementation.

For the fuller design-level assets, trust boundaries, and residual risks, see [THREAT_MODEL.md](THREAT_MODEL.md).

## Supported attack types

**No attack type is currently analyzed or detected by the implementation.** `GET /health` accepts no prompt content and returns no risk decision.

The following types are within the planned security scope only:

- Direct prompt injection in user-provided text.
- Attempts to override, reveal, or confuse higher-priority instructions.
- Evasion of simple lexical indicators through paraphrase or obfuscation.
- Indirect injection carried by retrieved documents or tool output.
- Benign quoted or educational security content that could otherwise be falsely flagged.

The list describes intended future evaluation and design coverage. It does not imply that any of these attacks are presently supported, blocked, or detected.

## Current implemented architecture

The current runtime consists of:

1. `backend/config.py`: immutable settings loaded from `CIPHER_APP_NAME`, `CIPHER_APP_VERSION`, and `CIPHER_ENVIRONMENT`, with defaults `CIPHER`, `0.1.0`, and `development`. Empty configured values raise `ConfigurationError` during app creation.
2. `backend/app.py`: a FastAPI application factory that accepts settings or loads them from the environment, sets application title/version, registers the health router, and maps unexpected exceptions to a generic HTTP 500 response.
3. `backend/api/health.py`: the `GET /health` route, which obtains settings from app state (failing cleanly with HTTP 500 if application state is not initialized) and returns process health metadata.
4. `backend/models/`: typed, immutable Pydantic models including `HealthResponse` and core domain contracts (`AnalysisInput`, `NormalizedContent`, `DetectorResult`, `RiskAssessment`, `Decision`, `PromptEnvelope`).
5. `backend/normalization/`: canonical normalization interface (`normalize_input`).
6. `backend/detector/`: abstract `BaseDetector` interface and packages for `rules/`, `embeddings/` (placeholder), and `classifier/` (placeholder).
7. `backend/risk/`: risk score fusion engine (`fuse_risk`).
8. `backend/policy/`: policy decision engine (`evaluate_decision`).
9. `backend/prompting/`: instruction/data separation envelope builder (`build_prompt_envelope`).
10. `backend/application/`: orchestration layer (`ApplicationService`) executing the complete end-to-end evaluation pipeline.

## Detection and decision components

Phase 1 foundation contracts and package boundaries are implemented. Production ML models and embedding backends remain deferred.

| Component | Current status | Responsibility |
|---|---|---|
| Normalization layer | Implemented (`backend/normalization/`) | Convert input to canonical analysis form (`NormalizedContent`) while preserving original text and source offsets. |
| Rule detector | Implemented (`backend/detector/rules/`) | Apply deterministic configured terms and structural indicator patterns (`RuleDetector`). |
| Semantic similarity detector | Boundary defined (`backend/detector/embeddings/`) | Deferred to Phase 5. Abstract port and placeholder defined. |
| ML classifier | Boundary defined (`backend/detector/classifier/`) | Deferred to Phase 6. Abstract port and placeholder defined. |
| Risk fusion engine | Implemented (`backend/risk/`) | Combine detector outputs into a `RiskAssessment` score and risk band while handling missing/unavailable detectors. |
| Decision engine | Implemented (`backend/policy/`) | Apply policy thresholds to generate a `Decision` (`allow`, `flag`, or `block`) with reason codes. |
| Structured prompt/data separation | Implemented (`backend/prompting/`) | Represent trusted instructions and untrusted data in typed distinct envelope fields (`PromptEnvelope`). |

No detection logic or detector dependencies are loaded by the current FastAPI app.

## API endpoints

### `GET /health`

Reports that the API process is responding. It does not check detector or downstream model readiness.

Example response (`200 OK`):

```json
{
  "status": "ok",
  "service": "CIPHER",
  "version": "0.1.0",
  "environment": "development"
}
```

The `service`, `version`, and `environment` fields reflect application settings and may differ when configured through environment variables or an injected `Settings` instance.

FastAPI's generated documentation endpoints are also enabled by default: `GET /docs`, `GET /redoc`, and `GET /openapi.json`. There is no analysis, classification, or scan endpoint.

## Request and response schemas

### Health request

There is no request body, query parameter, or custom authentication header defined for `GET /health`.

### Health response

The JSON response is validated by `HealthResponse`:

| Field | Type | Meaning |
|---|---|---|
| `status` | string | Fixed process status value `ok` for a successful health response. |
| `service` | string | Configured application name. |
| `version` | string | Configured application version. |
| `environment` | string | Configured environment label. |

Unexpected exceptions handled by the application produce `500 Internal Server Error` with the generic body `{"detail":"Internal server error"}`. Exception details are not returned. Invalid configuration fails while constructing the application; it is not represented as a health response.

No input-analysis request or response schema currently exists.

## Current limitations

- The service does not inspect prompts or external content and cannot identify attacks.
- No normalizer, rule set, embedding model/reference set, classifier, fusion logic, or policy thresholds are implemented.
- The health route reports process response only, not readiness of detectors or external services.
- No analysis endpoint, persistence, authentication, tenant isolation, rate limiting, or model/tool integration is implemented.
- No structured prompt builder is available; the host must preserve trusted/untrusted separation itself.
- A successful health response is not a security assessment or guarantee.

## Future indirect injection architecture

Indirect injection scanning is a future design boundary, not current behavior. The documented direction is to inspect each retrieved document, tool result, or other external content item independently before an agent uses it, retain provenance/source metadata, and return detector evidence to the integrating application's policy flow. The future scanner should reuse the same detector interfaces as direct input analysis rather than silently trusting externally sourced text.

The scan point (at ingestion, retrieval time, or synchronously before use), cache/freshness rules, failure behavior, and enforcement policy have not been selected or implemented. Even with scanning, the host application must separately authorize tool calls and data access and keep untrusted content distinct from trusted instructions.
