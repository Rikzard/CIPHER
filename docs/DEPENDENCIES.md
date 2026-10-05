# Backend dependencies

The repository selects Python 3.10 through `.python-version`. The root
`pyproject.toml` is the source of truth for the CIPHER development
environment, and `uv.lock` records the resolved dependency graph. The backend
runtime dependencies include FastAPI/Pydantic and FAISS. The default `dev`
dependency group contains `httpx`, required by FastAPI's `TestClient` in the
API tests, and Uvicorn, used by the development server. `uv run` includes that
default group and synchronizes the environment from the lockfile. The heavier
`sentence-transformers` package remains in the `embeddings` extra. Embedding
detection requires this extra in the development/runtime environment; install
the standard CIPHER development environment with:

```powershell
uv sync --extra embeddings
```

From the repository root, the canonical backend suite command is:

```powershell
uv run python -m unittest discover -s backend/tests -v
```

On a fresh checkout, install Python 3.10 and `uv`, run `uv sync --extra embeddings`,
then run that command. `uv` provisions the locked test and embedding package
dependencies. The embedding model weights and generated FAISS index remain
separate local assets; see
[EMBEDDINGS.md](EMBEDDINGS.md) for model/index provisioning. Model weights are
not downloaded by tests or API requests. Tests use deterministic test doubles.

Start the development server from the repository root with:

```powershell
uv run uvicorn backend.app:app --reload
```

This command expects the `embeddings` extra and the separately provisioned local
model/index to be available. Missing embedding resources are reported as an
unavailable detector; they are never fetched or rebuilt during a request.

`backend/detector/embeddings/requirements.txt` remains for the standalone
embedding provisioning commands. Keep its direct embedding dependency pins
aligned with the root project manifest when changing those packages.
