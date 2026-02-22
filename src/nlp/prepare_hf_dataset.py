"""Convert labeled sentences to HF token-classification dataset for CamemBERT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
from transformers import AutoTokenizer

MODEL_NAME = "camembert-base"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_labeled_csv(path: Path) -> pd.DataFrame:
    """Expected columns: sentence, labels (JSON array of {start,end,label})."""
    df = pd.read_csv(path)
    if "labels" not in df.columns or "sentence" not in df.columns:
        raise ValueError("CSV must contain 'sentence' and 'labels' columns.")
    return df


def align_labels(sentence: str, entities: List[Dict]) -> Dict:
    """Align char-level entities to token-level BIO labels."""
    tokens = TOKENIZER(sentence, return_offsets_mapping=True, truncation=True)
    offsets = tokens["offset_mapping"]
    labels = ["O"] * len(offsets)

    for ent in entities:
        start, end, label = int(ent["start"]), int(ent["end"]), ent["label"]
        for i, (s, e) in enumerate(offsets):
            if s >= end or e <= start:
                continue
            # token overlaps entity
            prefix = "B" if labels[i] == "O" and s == start else "I"
            labels[i] = f"{prefix}-{label}"
    tokens.pop("offset_mapping")
    tokens["labels"] = labels
    return tokens


def process(input_csv: Path, output_path: Path) -> None:
    df = load_labeled_csv(input_csv)
    encodings = []
    for _, row in df.iterrows():
        entities = json.loads(row["labels"])
        enc = align_labels(row["sentence"], entities)
        encodings.append(enc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in encodings:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(encodings)} records to {output_path}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="CSV with sentence,labels(json).")
    ap.add_argument("--output", type=Path, default=Path("data/dataset/hf_train.jsonl"))
    args = ap.parse_args()
    process(args.input, args.output)


if __name__ == "__main__":
    main()
