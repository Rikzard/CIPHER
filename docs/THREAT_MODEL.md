# CIPHER Threat Model

## Security objective

Help an integrating LLM application identify and make an explicit policy decision about prompt injection attempts in untrusted text, including direct user input and, in a later phase, indirect content from retrieval and tools. Preserve the distinction between trusted instructions and untrusted data throughout analysis and prompt preparation.

CIPHER reduces risk; it cannot establish that arbitrary content is safe. The host application remains responsible for authorization, tool access, secrets, and enforcement.

## Assets

- System and developer instructions and other trusted policy content.
- Credentials, private data, and application context accessible to the model or agent.
- Integrity of model actions, tool calls, and user-visible outputs.
- Provenance and integrity of retrieved documents and tool results.
- Detector configuration, reference examples, model artifacts, and policy thresholds.
- User-submitted content and any derived evidence, which may itself be sensitive.

## Trust boundaries

1. **Caller to gateway:** requests can be malformed, abusive, or intentionally adversarial.
2. **Untrusted text to analysis:** user text, retrieved material, and tool results can contain instructions disguised as data.
3. **Detector to decision:** each detector is fallible, can be unavailable, and may produce conflicting evidence.
4. **Gateway to integrating application:** the gateway response is advice until the caller enforces it.
5. **Trusted instructions to untrusted content:** serialization or concatenation can erase this distinction if the integration is careless.
6. **Evaluation data to runtime:** adversarial fixtures and labels must not be mistaken for production configuration or trusted prompts.

## Adversaries and capabilities

Assume an attacker can submit arbitrary text or cause an application to ingest attacker-controlled external content. The attacker may use obfuscation, paraphrase, multiple languages, quoted instructions, separator/context manipulations, or content intended to alter model or agent behavior. In an agent integration, the attacker may place instructions in documents or tool outputs and attempt to induce unauthorized disclosure or actions.

Depending on the deployment, an attacker may observe allow/block behavior and adapt inputs. Do not assume the detector is hidden, that a block decision ends the attack, or that a model will honor delimiters consistently.

## In-scope threats

- Direct prompt injection in user-supplied content.
- Attempts to override, reveal, or confuse higher-priority instructions.
- Evasion of simple lexical rules through wording changes or obfuscation.
- False positives that block benign quoted, educational, or security-analysis content.
- Conflicting detector outputs, detector failures, and policy misconfiguration.
- Indirect injection in retrieval results, documents, and tool outputs (future scanner boundary).
- Sensitive prompt leakage through logs, detector explanations, or API responses.
- Loss of trusted/untrusted separation when content is serialized into model context.

## Out of scope and residual risk

- Proving model behavior safe or eliminating prompt injection.
- Authorization of tool calls, user identity, or data access.
- Securing the LLM provider, host application, retrieval index, operating system, or network perimeter.
- Detecting every harmful, deceptive, or policy-violating request unrelated to prompt injection.
- Guaranteeing that an `allow` result is safe or that a `block` result is malicious.

Residual risks include novel attacks, multilingual and domain-specific evasion, training/reference data gaps, model behavior changes, incorrect calibration, and unsafe decisions in the calling application. Structured formatting can support separation but does not itself enforce authority.

## Defensive controls in the design

- Use multiple detector families with explicit availability and provenance rather than one brittle signal.
- Keep normalization, evidence generation, score fusion, and policy decisions independently versioned and testable.
- Make missing detector behavior explicit and fail according to configured policy rather than treating failure as a clean result.
- Keep raw text out of logs by default; expose bounded reason codes rather than full prompts or matched examples.
- Carry source provenance so future indirect scanning can apply per-source policy and evaluation.
- Represent trusted instructions and untrusted payload separately in prompt-building contracts.
- Evaluate benign and adversarial data, including context/separator/disruptor variants and agent-style scenarios.
- Require the integration to enforce decisions and separately authorize tools and data access.

## Security questions before deployment

- Which callers and tenants may use the gateway, and how are they authenticated and rate-limited?
- What are the data retention, redaction, and access rules for requests and evaluation traces?
- What should happen when a detector or dependency is unavailable: allow, flag, or block?
- Are thresholds and reference sets calibrated for the actual traffic and acceptable false-positive rate?
- How are rule, model, fusion, and policy versions rolled back and audited?
- How does the host enforce a decision on every model and agent path, including retries and tool calls?
