# Development Plan

This plan stages CIPHER from a minimal backend skeleton to an evaluated gateway. The current implementation includes configuration, a FastAPI app factory, `GET /health`, and its test; detection and analysis are not implemented. Each later phase should remain independently reviewable and avoid introducing a frontend or trained model ahead of its phase.

## Phase 0: Architecture and security documentation (complete)

- Record module boundaries, threat assumptions, research mapping, and target repository layout.
- Define the intended Python/FastAPI backend without adding runtime code or dependencies.
- Exit when the interfaces and security responsibilities are clear enough to implement incrementally.

## Phase 1: Backend skeleton and contracts (health/configuration foundation complete)

- Establish a minimal Python package layout and FastAPI application entry point. (Done.)
- Define typed request, response, normalized-content, detector-result, risk-assessment, decision, and prompt-envelope contracts.
- Add configuration loading and a narrow health/readiness surface. (`GET /health` and environment-backed settings are in place.)
- Keep storage, authentication systems, queues, and external services out until requirements demand them.

## Phase 2: Normalization and deterministic rules

- Implement input normalization with explicit versioning and source metadata.
- Add a deterministic rule-based detector for configurable banned terms and structural indicators.
- Return stable rule identifiers and bounded evidence; avoid logging raw input by default.
- Define behavior for malformed requests, detector errors, and unsupported inputs.

## Phase 3: Risk fusion and decision policy

- Define detector score semantics and missing/unavailable behavior.
- Add versioned score fusion and configurable `allow`, `flag`, and `block` thresholds.
- Keep the decision engine independent from detector algorithms.
- Record decision reason codes sufficient for callers and offline analysis without exposing sensitive content.

## Phase 4: Evaluation foundation

- Build versioned benign and adversarial fixtures with provenance and labels.
- Include direct attacks, HOUYI-inspired context/separator/disruptor variations, quoted/benign security content, and false-positive cases.
- Add reproducible offline evaluation for precision, recall, false-positive rate, false-negative rate, and latency where applicable.
- Use AgentDojo-inspired task/agent scenarios when an agent integration exists; do not present a small local set as benchmark equivalence.

## Phase 5: Embedding similarity (optional implementation phase)

- Select a local or hosted embedding backend only after privacy, retention, latency, and dependency requirements are decided.
- Version the embedding model and reference set; keep source examples private from API responses.
- Evaluate whether similarity adds measurable value over rules and establish calibration before enabling it in decisions.

## Phase 6: ML classifier (later; not part of this baseline)

- Define the labeled data governance and training/evaluation split before model selection.
- Choose a model/runtime and calibration method based on measured requirements.
- Track model version, training data lineage, thresholds, latency, and rollback process.
- Compare performance and failure modes against the simpler detector stack before deployment.

## Phase 7: Structured prompt/data integration

- Implement a typed representation of trusted instructions and untrusted content.
- Add an integration adapter for a selected model API format without allowing content to change trust level.
- Document that separators alone are not a security boundary; require host-side authorization and tool checks.

## Phase 8: Indirect injection scanning and agent evaluation

- Define provenance-aware scanner interfaces for retrieval documents, tool outputs, and other external sources.
- Decide whether scanning is synchronous, cached, or performed at ingestion, based on freshness and latency needs.
- Evaluate malicious and benign content through representative agent tasks, including attempted data disclosure and unauthorized tool actions.
- Keep scanner availability and enforcement behavior explicit in the integrating application's policy.

## Ongoing release criteria

- Security-sensitive behavior is covered by tests and reproducible evaluation as implementation proceeds.
- Each detector, fusion strategy, and policy configuration is versioned.
- Changes document false-positive/false-negative tradeoffs and failure-mode behavior.
- No raw prompt persistence is enabled by default; any retention is documented and explicitly configured.
- Dependencies are limited to those needed by the active phase and their operational impact is documented.
