"""Validate train_ner.jsonl and split into train/eval (80/20)."""

import json
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/dataset")
INPUT_FILE = DATA_DIR / "train_ner.jsonl"
TRAIN_FILE = DATA_DIR / "train_ner_train.jsonl"
EVAL_FILE = DATA_DIR / "train_ner_eval.jsonl"

LABEL_MAP = {
    0: "O",
    1: "B-DEPART",
    2: "I-DEPART",
    3: "B-ARRIVEE",
    4: "I-ARRIVEE",
    5: "B-DATE",
    6: "I-DATE",
    7: "B-VIA",
    8: "I-VIA",
}

REQUIRED_KEYS = {"id", "tokens", "ner_tags", "text"}


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def validate(samples: list[dict]) -> None:
    issues = []
    label_counts: Counter = Counter()
    token_lengths: list[int] = []

    for i, s in enumerate(samples):
        # Check keys
        missing = REQUIRED_KEYS - s.keys()
        if missing:
            issues.append(f"Sample {i} (id={s.get('id')}): missing keys {missing}")

        tokens = s.get("tokens", [])
        tags = s.get("ner_tags", [])

        # Check empty
        if not tokens:
            issues.append(f"Sample {i} (id={s.get('id')}): empty tokens")

        # Check length mismatch
        if len(tokens) != len(tags):
            issues.append(
                f"Sample {i} (id={s.get('id')}): token/tag length mismatch "
                f"({len(tokens)} vs {len(tags)})"
            )

        # Unknown labels
        for tag in tags:
            if tag not in LABEL_MAP:
                issues.append(
                    f"Sample {i} (id={s.get('id')}): unknown tag {tag}"
                )

        token_lengths.append(len(tokens))
        label_counts.update(tags)

    # Report
    print(f"Total samples: {len(samples)}")
    avg_len = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    print(f"Average token length: {avg_len:.1f}")

    print("\nLabel distribution:")
    for tag_id in sorted(LABEL_MAP.keys()):
        name = LABEL_MAP[tag_id]
        count = label_counts.get(tag_id, 0)
        print(f"  {name} ({tag_id}): {count}")

    if issues:
        print(f"\n{len(issues)} issues found:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("\nNo issues found.")


def split_and_write(samples: list[dict], seed: int = 42) -> None:
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.8)
    train = shuffled[:split_idx]
    eval_ = shuffled[split_idx:]

    for path, data in [(TRAIN_FILE, train), (EVAL_FILE, eval_)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} samples to {path}")


def verify_hf_loadable() -> None:
    from datasets import load_dataset

    for path in [TRAIN_FILE, EVAL_FILE]:
        ds = load_dataset("json", data_files=str(path))
        print(f"HuggingFace loaded {path}: {ds}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Validate dataset")
    print("=" * 60)
    samples = load_jsonl(INPUT_FILE)
    validate(samples)

    print("\n" + "=" * 60)
    print("Step 2: Split into train/eval (80/20, seed=42)")
    print("=" * 60)
    split_and_write(samples)

    print("\n" + "=" * 60)
    print("Step 3: Verify HuggingFace datasets can load files")
    print("=" * 60)
    verify_hf_loadable()

    print("\nDone!")
