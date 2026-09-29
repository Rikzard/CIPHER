"""Build the local FAISS index from the version-controlled JSONL examples.

Run with the project's embedding dependencies installed and a local
all-MiniLM-L6-v2 model directory configured through CIPHER_EMBEDDING_MODEL_PATH.
"""

from backend.detector.embeddings.embedding_detector import build_local_index


if __name__ == "__main__":
    print(build_local_index())
