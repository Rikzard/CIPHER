"""Offline fine-tuning entry point for the CIPHER binary classifier.

The base checkpoint is required as a local directory; this script never uses
the Hub or reads the frozen ``data/evaluation`` examples as training input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.detector.classifier.chunking import tokenize_overlapping_chunks
from backend.detector.classifier.ml_classifier import CHECKPOINT_NAME
from backend.training.validate_classifier_data import validate_classifier_data


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(
    *,
    base_model_path: Path,
    output_path: Path,
    data_dir: Path,
    frozen_evaluation_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 17,
) -> Path:
    """Fine-tune locally with a deterministic CPU loop and validation loss."""
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"Local base model directory missing: {base_model_path}")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    validate_classifier_data(data_dir, frozen_evaluation_dir)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, __version__ as transformers_version
        import tokenizers
    except ImportError as exc:
        raise RuntimeError("Install the optional classifier-train dependency group") from exc

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("Classifier chunking requires a fast tokenizer with offsets")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(base_model_path),
        local_files_only=True,
        trust_remote_code=False,
        num_labels=2,
        id2label={0: "benign", 1: "prompt_injection"},
        label2id={"benign": 0, "prompt_injection": 1},
    )
    model.to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_records = _read(data_dir / "train.jsonl")
    validation_records = _read(data_dir / "validation.jsonl")
    train_items = _tokenize_records(train_records, tokenizer)
    validation_items = _tokenize_records(validation_records, tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")

    output_path.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_items)))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [train_items[index] for index in order[start : start + batch_size]]
            input_ids, attention_mask, labels = _batch_tensors(batch, pad_token_id, torch)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()

        validation_loss = _mean_loss(model, validation_items, batch_size, pad_token_id, torch)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            model.save_pretrained(output_path, safe_serialization=True)
            tokenizer.save_pretrained(output_path)

    artifact_checksums = {
        path.relative_to(output_path).as_posix(): _sha256(path)
        for path in sorted(output_path.rglob("*"))
        if path.is_file() and path.name != "cipher_classifier_manifest.json"
    }
    if not artifact_checksums:
        raise RuntimeError("Training did not produce any model/tokenizer artifact files")
    manifest = {
        "artifact_schema_version": 1,
        "artifact_version": "distilbert-base-uncased-cipher-binary-v1",
        "model_name": "DistilBERT binary sequence classifier",
        "model_checkpoint": CHECKPOINT_NAME,
        "tokenizer_version": f"transformers={transformers_version};tokenizers={tokenizers.__version__};class={type(tokenizer).__name__}",
        "label_mapping": {"benign": 0, "prompt_injection": 1},
        "max_sequence_length": 512,
        "chunk_stride": 256,
        "aggregation": "max_chunk_probability",
        "finding_threshold": 0.5,
        "probability_calibrated": False,
        "training_dataset_version": json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["dataset_version"],
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_sha256": _sha256(data_dir / "train.jsonl"),
        "artifact_files_sha256": artifact_checksums,
        "validation_data_sha256": _sha256(data_dir / "validation.jsonl"),
        "training_config": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed},
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
    (output_path / "cipher_classifier_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def _tokenize_records(records: list[dict[str, Any]], tokenizer: Any) -> list[tuple[list[int], list[int], int]]:
    items: list[tuple[list[int], list[int], int]] = []
    for record in records:
        chunks, _ = tokenize_overlapping_chunks(record["text"], tokenizer, max_length=512, stride=256)
        items.extend((list(chunk.input_ids), list(chunk.attention_mask), int(record["label"])) for chunk in chunks)
    return items


def _batch_tensors(batch: list[tuple[list[int], list[int], int]], pad_id: int, torch: Any) -> tuple[Any, Any, Any]:
    ids = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    masks = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return input_ids, attention_mask, labels


def _mean_loss(model: Any, items: list[tuple[list[int], list[int], int]], batch_size: int, pad_id: int, torch: Any) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            input_ids, attention_mask, labels = _batch_tensors(batch, pad_id, torch)
            losses.append(float(model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.item()))
    if not losses:
        raise ValueError("Validation split produced no token chunks")
    return sum(losses) / len(losses)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=root / "models/classifier/base/distilbert-base-uncased")
    parser.add_argument("--output", type=Path, default=root / "models/classifier/active")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    result = train(
        base_model_path=args.base_model,
        output_path=args.output,
        data_dir=root / "data/classifier",
        frozen_evaluation_dir=root / "data/evaluation",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
