# CIPHER

CIPHER is a defense-in-depth prompt injection detection gateway for LLM applications. It is intended to assess untrusted user and application-provided content before that content reaches a model, and to provide an explicit policy decision to the integrating application.

## Project status

The repository contains architecture documentation and a minimal Python/FastAPI backend skeleton. `GET /health` is implemented. Detection, analysis endpoints, frontend, and integrations have not been implemented.

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
3. Independent detector interfaces produce evidence: deterministic rules, embedding similarity, and (in a later phase) an ML classifier.
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

The backend uses Python and FastAPI. An ASGI server such as Uvicorn can serve the app after runtime dependencies are installed. The `/health` test uses pytest and FastAPI's TestClient (which requires httpx). No detector or model dependencies are needed yet.

## Scope boundaries

- No ML model, trained weights, embedding service, frontend, or production scanner is included in this baseline.
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
