"""Fine-tune bert-base-uncased on the Ekman-6 (+neutral) label space.

This produces YOUR model — the one that lets you answer the research question
with numbers you generated, not borrowed. Fine-tuning wants a GPU: run this on
Google Colab (free GPU) as your risk table plans, then download the saved
`models/ekman-bert/` folder and point the API's MODEL_NAME at it.

Run (from backend/):
    python -m ml.train --epochs 3 --output models/ekman-bert
"""
from __future__ import annotations

import argparse

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from app.core.emotions import EKMAN
from ml.data import EKMAN_ID2LABEL, EKMAN_LABEL2ID, load_ekman_dataset
from ml.metrics import compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="bert-base-uncased")
    ap.add_argument("--output", default="models/ekman-bert")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    ds = load_ekman_dataset()

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    ds = ds.map(tokenize, batched=True)

    # num_labels + the id<->label maps teach the model (and the saved config)
    # exactly which output neuron means which emotion. The API reads these back.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(EKMAN),
        id2label=EKMAN_ID2LABEL,
        label2id=EKMAN_LABEL2ID,
    )

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        m = compute_metrics(labels, preds, EKMAN)
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    training_args = TrainingArguments(
        output_dir=f"{args.output}/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_metrics,
    )

    trainer.train()

    print("\nFinal test-set evaluation:")
    print(trainer.evaluate(ds["test"]))

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nSaved fine-tuned model -> {args.output}")
    print(f"Point the API at it:  MODEL_NAME={args.output}")


if __name__ == "__main__":
    main()
