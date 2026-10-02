# CIPHER

CIPHER is a defense-in-depth prompt injection detection gateway for LLM applications. It is intended to assess untrusted user and application-provided content before that content reaches a model, and to provide an explicit policy decision to the integrating application.

## Project status

The repository contains the Python/FastAPI backend, `GET /health`, `POST /analyze`, a rule detector, and an optional local semantic embedding detector. The embedding similarity threshold is a project calibration candidate and is not yet the production setting. See `docs/CIPHER_SPEC.md` for implementation status and limitations.

## Design goals

- Treat user input, retrieved content, tool results, and other externally sourced text as untrusted data.
- Keep independent detection signals separate so they can be evaluated and replaced independently.
- Make risk fusion and allow, flag, or block policy decisions explicit and inspectable.
- Preserve the boundary between trusted instructions and untrusted data when preparing model context.
- Evaluate defenses against direct and indirect injection scenarios, including benign inputs and false positives.
- Start with a small dependency footprint; add infrastructure only when an implementation phase requires it.

## Architecture

The intended request flow is:

1. A FastAPI adapter validates a request and assigns request metadata.
2. Input normalization creates a canonical analysis representation while retaining the original content for safe downstream use and audit references.
3. Independent detector interfaces produce evidence: deterministic rules and embedding similarity are used in `/analyze`; a standalone ML classifier is available for offline development but is not integrated into the API pipeline.
4. Risk fusion combines calibrated detector outputs into an explainable assessment.
5. A decision engine applies configured thresholds and policy to return `allow`, `flag`, or `block`.
6. If the application proceeds, a structured prompt/data boundary carries trusted instructions separately from untrusted content.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module boundaries and contracts, [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for security assumptions, and [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for implementation phases.

## Research basis

The design is informed by the following lines of work:

- HOUYI: black-box prompt injection, including context, separator, and disruptor concepts.
- *Prompt Injection Detection in LLM Integrated Applications*: banned-term, embedding, classifier, and decision-layer approaches.
- *Prompt Injection Attacks in LLMs and AI Agent Systems*: defense-in-depth and indirect injection.
- StruQ: structured separation of instructions and untrusted data.
- AgentDojo: security evaluation in adversarial agent environments.

These works inform design questions and evaluation scenarios; the architecture does not claim to reproduce or validate their results.

## Development

The backend development environment is declared in the root `pyproject.toml` and locked in `uv.lock`. From the repository root, run the canonical backend test suite with:

```powershell
uv run python -m unittest discover -s backend/tests -v
```

`uv run` installs/synchronizes the locked backend dependencies and the default development group, including the FastAPI test client dependency. See [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) for the dependency layout and [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md) for local model and FAISS setup.

## Scope boundaries

- The standalone ML classifier and its development workflow are implemented, but it is not integrated into `/analyze`; no trained checkpoint is committed. No frontend or indirect-content scanner is implemented.
- Detection is advisory/policy input, not a proof that content is safe.
- Integrating applications remain responsible for authorization, tool permissions, secret handling, and enforcing CIPHER decisions.

## Repository layout (target)

```text
backend/       FastAPI adapter, configuration, models, detector boundary, and tests
docs/          architecture, threat model, and phased plan
evaluation/    datasets, scenarios, runners, and reports
tests/         unit, contract, and integration tests
data/          versioned benign/adversarial fixtures only; no secrets
frontend/      intentionally out of scope for the initial backend phases
```
