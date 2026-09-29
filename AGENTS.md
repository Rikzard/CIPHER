# Repository guidance

## Scope and architecture

- CIPHER is a defense-in-depth prompt injection detection gateway. The planned backend uses Python and FastAPI.
- Follow the module boundaries and request flow in `docs/ARCHITECTURE.md`; keep transport, detection, fusion, policy decisions, prompt construction, and evaluation responsibilities separate.
- Treat user-provided text, retrieved material, and tool outputs as untrusted. Preserve trusted-instruction and untrusted-data separation.
- Do not claim detection guarantees. Surface uncertainty and detector evidence in a controlled, non-sensitive form.

## Change discipline

- Keep changes focused on the requested scope. Do not add a frontend, ML model, or unrelated infrastructure unless explicitly requested.
- Avoid unnecessary dependencies. Explain any new runtime or development dependency in the relevant change documentation.
- Keep secrets, real credentials, and sensitive production content out of source control and evaluation fixtures.
- Do not log raw prompts by default. If future debugging or evaluation requires content capture, define retention, access, and redaction behavior first.
- Keep documentation aligned with the implemented interfaces and clearly label planned or deferred components.

## Validation

- Do not add or run tests unless the user requests testing or verification.
- When implementation work is requested, report the files changed and the checks actually performed. Never imply that a planned component exists when it does not.
