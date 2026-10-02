# Backend dependencies

The repository selects Python 3.10 through `.python-version`. The root
`pyproject.toml` is the source of truth for the CIPHER development
environment, and `uv.lock` records the resolved dependency graph. The backend
runtime dependencies include FastAPI/Pydantic and FAISS. The default `dev`
dependency group contains `httpx`, required by FastAPI's `TestClient` in the
API tests. `uv run` includes that default group and synchronizes the
environment from the lockfile. The heavier `sentence-transformers` package is
in the `embeddings` extra because backend tests use deterministic doubles and
do not need model inference. Install it for local embedding inference with
`uv sync --extra embeddings`.

From the repository root, the canonical backend suite command is:

```powershell
uv run python -m unittest discover -s backend/tests -v
```

On a fresh checkout, install Python 3.10 and `uv`, then run that command. `uv`
provisions the locked test dependencies as needed. The embedding model weights
and generated FAISS index remain separate local assets; see
[EMBEDDINGS.md](EMBEDDINGS.md) for model/index provisioning. Model weights are
not downloaded by tests automatically; tests that need embeddings use the
repository's locally provisioned model where available or deterministic test
doubles.

`backend/detector/embeddings/requirements.txt` remains for the standalone
embedding provisioning commands. Keep its direct embedding dependency pins
aligned with the root project manifest when changing those packages.
