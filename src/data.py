"""Carga del conjunto de evaluación.

1) Preferido: v1adam/Comparison_Dataset de Hugging Face (Data Availability, p. 21).
   El artículo no documenta los nombres de columna, así que se detectan automáticamente;
   si la detección falla, ajuste TEXT_COLS / LABEL_COLS / SUBTYPE_COLS.
2) Respaldo: data/simulado_120.csv (misma estructura, datos simulados).
"""
from pathlib import Path

import pandas as pd

from .config import HF_EVAL_DATASET

TEXT_COLS = ("prompt", "text", "input", "query", "content")
LABEL_COLS = ("label", "category", "class", "type")
SUBTYPE_COLS = ("subtype", "subcategory", "benign_type", "sub_label", "subclass")
SIMULATED = Path(__file__).resolve().parents[1] / "data" / "simulado_120.csv"


def _norm_label(v) -> str:
    s = str(v).strip().lower()
    for c in ("benign", "sensitive", "jailbreak"):
        if c in s:
            return c
    return {"0": "benign", "1": "sensitive", "2": "jailbreak"}.get(s, s)


def _norm_subtype(v, label) -> str:
    s = str(v).strip().lower()
    if label != "benign":
        return "-"
    return "hard_negative" if "hard" in s else "general"


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    tc = next((cols[c] for c in TEXT_COLS if c in cols), None)
    lc = next((cols[c] for c in LABEL_COLS if c in cols), None)
    if tc is None or lc is None:
        raise ValueError(f"No se reconocen las columnas {list(df.columns)}. Ajuste TEXT_COLS/LABEL_COLS en src/data.py")
    sc = next((cols[c] for c in SUBTYPE_COLS if c in cols), None)
    out = pd.DataFrame({"prompt": df[tc].astype(str), "label": df[lc].map(_norm_label)})
    out["subtype"] = [_norm_subtype(df[sc].iloc[i] if sc else df[lc].iloc[i], l) for i, l in enumerate(out["label"])]
    return out.reset_index(drop=True)


def load_eval(source: str = "auto") -> tuple[pd.DataFrame, str]:
    if source in ("auto", "hf"):
        try:
            from datasets import load_dataset
            ds = load_dataset(HF_EVAL_DATASET)
            split = "test" if "test" in ds else list(ds.keys())[0]
            df = ds[split].to_pandas()
            print(f"[data] {HF_EVAL_DATASET} ({split}): {len(df)} filas, columnas {list(df.columns)}")
            return normalise(df), f"Hugging Face: {HF_EVAL_DATASET}"
        except Exception as e:
            if source == "hf":
                raise
            print(f"[data] No se pudo cargar {HF_EVAL_DATASET} ({e}). Se usa el conjunto simulado.")
    return normalise(pd.read_csv(SIMULATED)), "Simulado: data/simulado_120.csv"
