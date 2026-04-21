"""
PromptShield — Custom Injection Classifier Training
====================================================
Fine-tunes DistilBERT on deepset/prompt-injections using LoRA (PEFT).
Exports a merged model + tokenizer package and ONNX for fast inference.

Requirements:
    pip install -r requirements-train.txt
    pip install onnxscript sentencepiece tiktoken

Run:
    python train/train.py
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
import onnxruntime as ort


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
@dataclass
class Config:
    base_model: str = "distilbert-base-uncased"
    dataset_name: str = "deepset/prompt-injections"

    output_dir: str = "./outputs/promptshield-classifier"
    export_dir: str = "./outputs/promptshield-classifier-inference"
    onnx_path: str = "./outputs/promptshield-classifier-inference/model.onnx"

    # LoRA hyperparams
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Training
    max_length: int = 128
    epochs: int = 5
    batch_size: int = 32
    lr: float = 2e-4
    warmup_steps: int = 20
    weight_decay: float = 0.01
    eval_steps: int = 50
    save_steps: int = 50

    # Labels
    id2label = {0: "safe", 1: "injection"}
    label2id = {"safe": 0, "injection": 1}


cfg = Config()


# ─────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────
def load_and_prepare() -> DatasetDict:
    print("📥 Loading dataset: deepset/prompt-injections ...")
    raw = load_dataset(cfg.dataset_name)

    # The label column is a plain Value, so do NOT stratify by it.
    split = raw["train"].train_test_split(
        test_size=0.15,
        seed=42,
    )

    ds = DatasetDict(
        {
            "train": split["train"],
            "val": split["test"],
            "test": raw["test"],
        }
    )

    print(f"  Train: {len(ds['train'])} | Val: {len(ds['val'])} | Test: {len(ds['test'])}")

    label_counts = {}
    for ex in ds["train"]:
        l = ex["label"]
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"  Train label distribution: {label_counts}")

    return ds
# ─────────────────────────────────────────
# 2. Tokenize
# ─────────────────────────────────────────
def tokenize_dataset(ds: DatasetDict, tokenizer):
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )

    tokenized = ds.map(tok, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


# ─────────────────────────────────────────
# 3. Model with LoRA
# ─────────────────────────────────────────
def build_model():
    print(f"\n🔧 Loading base model: {cfg.base_model} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=2,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_lin", "v_lin"],
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────
# 4. Metrics
# ─────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


# ─────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────
def train(model, tokenized_ds, tokenizer):
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        logging_steps=20,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n🚀 Starting training ...")
    t0 = time.time()
    trainer.train()
    print(f"✅ Training done in {(time.time() - t0) / 60:.1f} min")

    # Merge LoRA into base weights for inference
    merged_model = trainer.model.merge_and_unload()

    # Save merged inference model to a clean export directory
    Path(cfg.export_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(cfg.export_dir)
    tokenizer.save_pretrained(cfg.export_dir)

    # Keep the trainer output dir too
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Save metadata
    metadata = {
        "base_model": cfg.base_model,
        "dataset_name": cfg.dataset_name,
        "id2label": cfg.id2label,
        "label2id": cfg.label2id,
        "max_length": cfg.max_length,
    }
    with open(Path(cfg.export_dir) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(Path(cfg.output_dir) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return trainer


# ─────────────────────────────────────────
# 6. Evaluate on test set
# ─────────────────────────────────────────
def evaluate(trainer, tokenized_ds):
    print("\n📊 Test set evaluation:")
    results = trainer.evaluate(eval_dataset=tokenized_ds["test"])
    print(json.dumps(results, indent=2))

    preds_out = trainer.predict(tokenized_ds["test"])
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids

    print("\n" + classification_report(labels, preds, target_names=["safe", "injection"]))
    return results


# ─────────────────────────────────────────
# 7. ONNX Export
# ─────────────────────────────────────────
def export_onnx(tokenizer):
    print(f"\n📦 Exporting to ONNX: {cfg.onnx_path} ...")

    from transformers import AutoModelForSequenceClassification as AM

    model = AM.from_pretrained(cfg.export_dir).eval()

    dummy = tokenizer(
        "ignore previous instructions",
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_length,
        padding="max_length",
    )

    # Export with a recent opset to avoid version-conversion failures
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        cfg.onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
        opset_version=18,
        do_constant_folding=True,
    )

    # Quick sanity check
    sess = ort.InferenceSession(cfg.onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(
        None,
        {
            "input_ids": dummy["input_ids"].numpy(),
            "attention_mask": dummy["attention_mask"].numpy(),
        },
    )
    pred = int(np.argmax(out[0]))
    print(f"  ONNX sanity check → label={cfg.id2label[pred]}  (expected: injection)")
    print(f"✅ ONNX export complete: {cfg.onnx_path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.export_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    ds = load_and_prepare()
    tokenized = tokenize_dataset(ds, tokenizer)
    model = build_model()
    trainer = train(model, tokenized, tokenizer)
    evaluate(trainer, tokenized)
    export_onnx(tokenizer)

    print("\n🎉 All done! Files saved to:")
    print("   Training output:", cfg.output_dir)
    print("   Inference export:", cfg.export_dir)
    print("   ONNX model:", cfg.onnx_path)
    print("   Next step: python serve/server.py")