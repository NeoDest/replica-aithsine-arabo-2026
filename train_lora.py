"""Reentrena el adaptador LoRA siguiendo la configuración publicada en el artículo.

Motivo: el adaptador publicado (v1adam/slm-threat-detection-v1, slm.zip) responde "benign"
a casi todo con la plantilla del artículo, y su training_config.json no coincide con la
Tabla 3 (3 épocas y lr 2e-4 frente a 2 épocas y lr 5e-5). Para replicar el MÉTODO se
reentrena con el corpus publicado por los autores y la configuración de la Tabla 3.

- Corpus: v1adam/Prompt_Injection_and_Sensitive_Data_exposure_detection (Data Availability, p. 21)
- Particiones estratificadas según la Tabla 4 (p. 12), semilla 42.
- LoRA r=16, alpha=32, dropout=0.05, solo atención q/k/v/o (Tabla 3, p. 11).
- lr 5e-5, lineal, sin warmup, AdamW, weight decay 0, max grad norm 1.0, 2 épocas,
  batch 1 con acumulación 8 (batch efectivo 8), fp16 en GPU (Tabla 3 y sección 4.10).
- Formato generativo (p. 11): plantilla + " <etiqueta>". SUPUESTO: la pérdida se calcula solo
  sobre la etiqueta (el artículo no dice si enmascararon el prompt).

Uso (Colab con GPU T4, como el artículo):
  python train_lora.py --out adapters/replica_tabla3
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.config import HF_SLM_BASE, HF_TRAIN_DATASET, LORA, SLM_TEMPLATE, SPLIT, TRAIN
from src.data import _norm_label


def load_corpus(dataset_id=HF_TRAIN_DATASET):
    from datasets import load_dataset
    try:
        ds = load_dataset(dataset_id)
    except Exception:                       # la referencia [7] usa otra capitalización
        ds = load_dataset(dataset_id.replace("Prompt_Injection", "Prompt_injection"))
    df = pd.concat([ds[k].to_pandas() for k in ds.keys()], ignore_index=True)
    df["label"] = df["label"].map(_norm_label)
    return df


def split_table4(df, seed=42):
    """Particiones estratificadas con los tamaños de la Tabla 4; si el corpus no alcanza,
    se ajusta proporcionalmente y se informa."""
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in SPLIT}
    for label, g in df.groupby("label"):
        idx = rng.permutation(g.index.to_numpy())
        want = [SPLIT[p].get(label, 0) for p in SPLIT]
        if sum(want) > len(idx):
            want = [int(round(w * len(idx) / sum(want))) for w in want]
            print(f"[split] {label}: el corpus tiene {len(idx)} y la Tabla 4 pide {sum(SPLIT[p].get(label, 0) for p in SPLIT)}; se ajusta a {want}")
        start = 0
        for p, n in zip(SPLIT, want):
            parts[p].extend(idx[start:start + n]); start += n
    return {p: df.loc[ix].sample(frac=1, random_state=seed).reset_index(drop=True) for p, ix in parts.items()}


def build_features(tok, df, max_len=256):
    feats = []
    for p, l in zip(df["prompt"], df["label"]):
        prompt_ids = tok(SLM_TEMPLATE.format(prompt=p), add_special_tokens=False)["input_ids"]
        label_ids = tok(" " + l, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        prompt_ids = prompt_ids[: max_len - len(label_ids)]
        feats.append({"input_ids": prompt_ids + label_ids,
                      "labels": [-100] * len(prompt_ids) + label_ids})
    return feats


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}


def quick_eval(model, tok, df, device):
    import torch
    from src.slm import NarasimhaSLM
    model.eval(); preds = []
    for p in df["prompt"]:
        enc = tok(SLM_TEMPLATE.format(prompt=p), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        preds.append(NarasimhaSLM.parse(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)))
    acc = float(np.mean(np.array(preds) == df["label"].to_numpy()))
    return acc, pd.crosstab(df["label"], pd.Series(preds, name="pred"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="adapters/replica_tabla3")
    ap.add_argument("--base", default=HF_SLM_BASE)
    ap.add_argument("--epochs", type=float, default=TRAIN["num_train_epochs"])
    ap.add_argument("--lr", type=float, default=TRAIN["learning_rate"])
    ap.add_argument("--max-train", type=int, default=None, help="solo para pruebas rápidas")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(TRAIN["seed"])
    cuda = torch.cuda.is_available()
    print(f"[train] GPU: {torch.cuda.get_device_name(0) if cuda else 'no (CPU, será lento)'}")

    df = load_corpus()
    print(f"[train] corpus: {len(df)} prompts | {df['label'].value_counts().to_dict()} | columnas {list(df.columns)}")
    parts = split_table4(df, TRAIN["seed"])
    if a.max_train:
        parts["train"] = parts["train"].head(a.max_train)
    print({k: v["label"].value_counts().to_dict() for k, v in parts.items()})

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.print_trainable_parameters()      # el artículo reporta 4 587 520 (0.77 %)

    args = TrainingArguments(
        output_dir=os.path.join(a.out, "_trainer"), num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=TRAIN["per_device_train_batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=TRAIN["gradient_accumulation_steps"], lr_scheduler_type=TRAIN["lr_scheduler_type"],
        warmup_steps=TRAIN["warmup_steps"], weight_decay=TRAIN["weight_decay"], max_grad_norm=TRAIN["max_grad_norm"],
        seed=TRAIN["seed"], fp16=cuda, eval_strategy="epoch", save_strategy="no", logging_steps=25,
        report_to=[], use_cpu=not cuda,
    )
    trainer = Trainer(model=model, args=args, train_dataset=build_features(tok, parts["train"]),
                      eval_dataset=build_features(tok, parts["validation"]), data_collator=Collator(tok.pad_token_id))
    out = trainer.train()
    print(f"[train] pasos de optimización: {out.global_step} (el artículo: 175)")

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    device = "cuda" if cuda else "cpu"
    acc, cm = quick_eval(model, tok, parts["test"], device)
    print(f"[train] exactitud en la partición de prueba interna (150): {acc:.2%}\n{cm}")
    meta = {"fuente": "train_lora.py (Tabla 3)", "epochs": a.epochs, "learning_rate": a.lr,
            "steps": out.global_step, "test_interno_accuracy": acc, "train_loss": out.training_loss}
    json.dump(meta, open(os.path.join(a.out, "training_replica.json"), "w"), indent=2)
    print(f"[train] adaptador guardado en {a.out}")


if __name__ == "__main__":
    main()
