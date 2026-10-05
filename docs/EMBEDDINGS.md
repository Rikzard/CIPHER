# Local embedding resources

The semantic detector is a local component used alongside `RuleDetector`. Its Python runtime requires the `embeddings` extra in the development/runtime environment. Project dependencies are declared in the root `pyproject.toml` and locked in `uv.lock`; `backend/detector/embeddings/requirements.txt` remains available for standalone provisioning workflows. From the repository root, install the development environment with:

```powershell
uv sync --extra embeddings
```

Start CIPHER with:

```powershell
uv run uvicorn backend.app:app --reload
```

The detector uses a local Sentence Transformer, the version-controlled JSONL dataset, and a generated local FAISS index. It does not install Python dependencies, download model files, or rebuild the index during API startup or requests. The current model is `sentence-transformers/all-MiniLM-L6-v2`.

Run the backend test suite with the canonical command from the repository root:

```powershell
uv run python -m unittest discover -s backend/tests -v
```

## 1. Install/provision the model once

From the repository root, download the model into the ignored `models/` directory:

```powershell
uv run hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
```

This is the only network step. Keep the resulting model directory local; do not commit model weights. The build and validation commands below load only from that local directory.

## 2. Configure local paths

The defaults assume commands run from the repository root:

| Environment variable | Default | Purpose |
|---|---|---|
| `CIPHER_EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Model identifier recorded in metadata and checked against runtime configuration. |
| `CIPHER_EMBEDDING_MODEL_PATH` | `models/all-MiniLM-L6-v2` | Existing local Sentence Transformer directory. |
| `CIPHER_EMBEDDING_INDEX_PATH` | `data/embeddings/injection_examples.faiss` | Generated FAISS index location. |
| `CIPHER_EMBEDDING_DATASET_PATH` | `data/embedding_examples.jsonl` | Version-controlled reference examples. |
| `CIPHER_EMBEDDING_MANIFEST_PATH` | `data/embedding_examples.manifest.json` | Dataset version and filename. |
| `CIPHER_EMBEDDING_THRESHOLD` | `0.65` | Initial, uncalibrated cosine-similarity finding threshold. |
| `CIPHER_EMBEDDING_TOP_K` | `3` | Number of nearest reference examples returned. |

For PowerShell, set custom paths in the current session before running the commands:

```powershell
$env:CIPHER_EMBEDDING_MODEL_PATH = "D:\local-models\all-MiniLM-L6-v2"
$env:CIPHER_EMBEDDING_INDEX_PATH = "D:\cipher-data\injection_examples.faiss"
```

The model name and path used to build an index must match those used later by the application. The dataset manifest and dataset checksum must also match the generated index metadata.

## 3. Build and validate the index

Build (or safely replace) the local index and its metadata:

```powershell
uv run --offline python -m backend.detector.embeddings.provision build
```

Validate the model, dataset, index, metadata, dimensions, and a sample nearest-neighbor query:

```powershell
uv run --offline python -m backend.detector.embeddings.provision validate
```

Both commands fail with a concise error and a nonzero exit status if a required local resource is missing or incompatible. `build` loads the model from the configured local directory, canonicalizes the dataset examples with CIPHER's normalizer, generates embeddings, creates a cosine-similarity FAISS index, writes metadata, reloads the saved resources, and validates them. Re-running `build` with the same model, dataset, configuration, and dependency versions produces the same metadata and replaces the generated index.

Provisioning metadata is deterministic JSON containing:

- metadata schema version;
- dataset version and normalized-line-ending SHA-256 checksum;
- embedding model name and configured path descriptor;
- embedding dimension and `cosine` similarity metric;
- number of indexed examples and their IDs/categories/text;
- SHA-256 checksum of the serialized FAISS index.

## Dataset and generated files

`data/embedding_examples.jsonl` is the source dataset, with examples grouped under `instruction_override`, `system_prompt_extraction`, `role_manipulation`, `task_redirection`, `context_manipulation`, `delimiter_manipulation`, and `obfuscation`. `data/embedding_examples.manifest.json` identifies dataset version `1.0.0`. Update the manifest version when intentionally revising the corpus.

Generated resources are excluded by the repository `.gitignore`: the local `models/` directory, `data/embeddings/*.faiss`, and its `*.faiss.json` metadata sidecar. Do not commit model weights, generated FAISS indexes, or their metadata. Commit intentional changes to the dataset and its manifest.

The current similarity threshold remains uncalibrated. Provisioning verifies consistency and queryability; it does not establish detection quality or calibrate a decision threshold. Threshold evaluation remains part of the evaluation phase.

## Labeled evaluation data

Threshold evaluation uses the separate project datasets under `data/evaluation/`, not the malicious-only FAISS reference corpus. `calibration.jsonl` includes benign hard negatives and malicious examples for score analysis; `test.jsonl` is held out and must not be used to choose a threshold. Validate schema and exact/canonical separation (using CIPHER's normalizer) from the repository root with:

```powershell
uv run --offline python -m backend.evaluation.datasets
```

The fixtures are small and hand-authored, not representative of all real-world prompt injections. The threshold remains uncalibrated; see [EVALUATION.md](EVALUATION.md) for the planned Step 13C analysis and limitations.
