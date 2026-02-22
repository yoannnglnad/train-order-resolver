"""Evaluate CamemBERT NER model with per-entity precision/recall/F1 metrics.

Usage:
  python scripts/evaluate_ner_metrics.py \
    --model data/models/camembert-ner \
    --eval data/dataset/train_ner_eval.jsonl \
    --out data/logs/eval_results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

LABEL_LIST = [
    "O",
    "B-DEPART",
    "I-DEPART",
    "B-ARRIVEE",
    "I-ARRIVEE",
    "B-VIA",
    "I-VIA",
    "B-DATE",
    "I-DATE",
]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
ENTITY_TYPES = ["DEPART", "ARRIVEE", "VIA", "DATE"]


def load_eval_dataset(path: Path) -> List[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def tags_to_spans(tag_ids: List[int]) -> List[Tuple[str, int, int]]:
    """Convert a sequence of tag IDs to (entity_type, start, end) spans."""
    spans = []
    current_type = None
    start = -1
    for i, tid in enumerate(tag_ids):
        label = ID2LABEL.get(tid, "O")
        if label.startswith("B-"):
            if current_type is not None:
                spans.append((current_type, start, i))
            current_type = label[2:]
            start = i
        elif label.startswith("I-"):
            etype = label[2:]
            if current_type != etype:
                if current_type is not None:
                    spans.append((current_type, start, i))
                current_type = None
        else:
            if current_type is not None:
                spans.append((current_type, start, i))
            current_type = None
    if current_type is not None:
        spans.append((current_type, start, len(tag_ids)))
    return spans


def compute_metrics(
    gold_spans_all: List[List[Tuple[str, int, int]]],
    pred_spans_all: List[List[Tuple[str, int, int]]],
) -> Dict:
    """Compute per-entity-type precision, recall, F1 and overall token accuracy."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for gold_spans, pred_spans in zip(gold_spans_all, pred_spans_all):
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)
        for etype in ENTITY_TYPES:
            gold_etype = {s for s in gold_set if s[0] == etype}
            pred_etype = {s for s in pred_set if s[0] == etype}
            tp[etype] += len(gold_etype & pred_etype)
            fp[etype] += len(pred_etype - gold_etype)
            fn[etype] += len(gold_etype - pred_etype)

    results = {}
    for etype in ENTITY_TYPES:
        p = tp[etype] / (tp[etype] + fp[etype]) if (tp[etype] + fp[etype]) > 0 else 0.0
        r = tp[etype] / (tp[etype] + fn[etype]) if (tp[etype] + fn[etype]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[etype] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    # Micro-average
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    results["micro_avg"] = {
        "precision": round(micro_p, 4),
        "recall": round(micro_r, 4),
        "f1": round(micro_f1, 4),
    }

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("data/models/camembert-ner"))
    ap.add_argument("--eval", type=Path, default=Path("data/dataset/train_ner_eval.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/logs/eval_results.json"))
    args = ap.parse_args()

    print(f"Loading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval()

    print(f"Loading eval dataset from {args.eval} ...")
    samples = load_eval_dataset(args.eval)
    print(f"  {len(samples)} samples loaded")

    gold_spans_all = []
    pred_spans_all = []
    total_tokens = 0
    correct_tokens = 0

    for sample in samples:
        tokens = sample["tokens"]
        gold_tags = sample["ner_tags"]

        # Tokenize
        encoded = tokenizer(
            tokens, is_split_into_words=True, truncation=True, return_tensors="pt"
        )
        word_ids = encoded.word_ids(batch_index=0)

        with torch.no_grad():
            logits = model(**encoded).logits[0]
        pred_ids = logits.argmax(-1).tolist()

        # Map subword predictions back to word-level (use first subword label)
        word_pred_tags = []
        seen_words = set()
        for i, wid in enumerate(word_ids):
            if wid is not None and wid not in seen_words:
                seen_words.add(wid)
                word_pred_tags.append(pred_ids[i])

        # Truncate to same length as gold
        min_len = min(len(gold_tags), len(word_pred_tags))
        gold_tags_trunc = gold_tags[:min_len]
        word_pred_tags_trunc = word_pred_tags[:min_len]

        # Token accuracy
        for g, p in zip(gold_tags_trunc, word_pred_tags_trunc):
            total_tokens += 1
            if g == p:
                correct_tokens += 1

        # Span-level metrics
        gold_spans = tags_to_spans(gold_tags_trunc)
        pred_spans = tags_to_spans(word_pred_tags_trunc)
        gold_spans_all.append(gold_spans)
        pred_spans_all.append(pred_spans)

    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    entity_metrics = compute_metrics(gold_spans_all, pred_spans_all)

    # Print report
    print("\n" + "=" * 60)
    print("NER Evaluation Report")
    print("=" * 60)
    print(f"\nToken Accuracy: {token_accuracy:.4f} ({correct_tokens}/{total_tokens})")
    print(f"\n{'Entity':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 44)
    for etype in ENTITY_TYPES:
        m = entity_metrics[etype]
        print(f"{etype:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print("-" * 44)
    m = entity_metrics["micro_avg"]
    print(f"{'MICRO AVG':<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print("=" * 60)

    # Check thresholds
    warnings = []
    for etype in ["DEPART", "ARRIVEE"]:
        if entity_metrics[etype]["f1"] < 0.8:
            warnings.append(
                f"WARNING: {etype} F1 = {entity_metrics[etype]['f1']:.4f} < 0.8. "
                f"Consider: more training data, more epochs, or lower learning rate."
            )
    if warnings:
        print()
        for w in warnings:
            print(w)

    # Save results
    output = {
        "token_accuracy": round(token_accuracy, 4),
        "entity_metrics": entity_metrics,
        "num_samples": len(samples),
        "warnings": warnings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
