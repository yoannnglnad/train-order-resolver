"""Fine-tune CamemBERT for token classification (DEPART/ARRIVEE/VIA/DATE)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "camembert-base"
LABEL_LIST = ["O", "B-DEPART", "I-DEPART", "B-ARRIVEE", "I-ARRIVEE", "B-VIA", "I-VIA", "B-DATE", "I-DATE"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=False)
    ap.add_argument("--out", type=Path, default=Path("data/models/camembert-ner"))
    ap.add_argument("--model_id", type=str, default="almanach/camembert-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    ds_train = load_dataset("json", data_files=str(args.train))["train"]
    ds_eval = load_dataset("json", data_files=str(args.eval))["train"] if args.eval else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    def tokenize_and_align(batch):
        tokenized = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
        aligned_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous_word = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(labels[word_id])
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    ds_train = ds_train.map(tokenize_and_align, batched=True)
    if ds_eval:
        ds_eval = ds_eval.map(tokenize_and_align, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_id, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )
    collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.out),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        push_to_hub=False,
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        # simple token accuracy
        correct = (preds == labels).astype(float)
        return {"token_accuracy": correct.sum() / correct.size}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if ds_eval else None,
    )
    trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
