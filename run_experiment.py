"""Experimento de réplica de Ait Hsine y Arabo (2026), sección 5.

Igual que en el artículo (sección 6.6, p. 19): se ejecutan TODOS los componentes sobre
TODOS los prompts y, después, se aplica la regla del filtro a las decisiones registradas
para obtener el resultado de la arquitectura con enrutamiento (sección 5.4, p. 15).

Uso:
  python run_experiment.py --data auto --slm --deberta          # réplica en CPU
  python run_experiment.py --data sim --no-slm                  # prueba rápida sin modelos
  OPENROUTER_API_KEY=... python run_experiment.py --llm         # incluye la capa externa (opcional, de pago)
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BOOTSTRAP_RESAMPLES, HF_DEBERTA_BASELINE, PAPER_RESULTS, PAPER_ROUTING, SEED
from src.data import load_eval
from src.external_llm import ExternalLLM
from src.heuristics import HeuristicEngine
from src.metrics import binary_metrics, bootstrap_accuracy_ci, per_category, wilson_ci
from src.rules import RuleEngine


def timed(fn, *a):
    t = time.perf_counter(); r = fn(*a); return r, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["auto", "hf", "sim"], default="auto")
    ap.add_argument("--slm", dest="slm", action="store_true", default=True)
    ap.add_argument("--no-slm", dest="slm", action="store_false")
    ap.add_argument("--adapter", default=None,
                    help="adaptador LoRA: ID de Hugging Face (por defecto el publicado) o carpeta local de train_lora.py")
    ap.add_argument("--deberta", action="store_true")
    ap.add_argument("--llm", action="store_true", help="usa OpenRouter (requiere OPENROUTER_API_KEY)")
    ap.add_argument("--rules-profile", choices=["full", "paper"], default="paper",
                    help="'paper' excluye los formatos que el motor de los autores no detectó (p. 18)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df, source = load_eval(a.data)
    if a.limit:
        df = df.groupby("label", group_keys=False).head(a.limit)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    y_true = (df["label"] != "benign").to_numpy()

    rules, heur = RuleEngine(profile=a.rules_profile), HeuristicEngine()
    slm = deb = None
    if a.slm:
        from src.slm import NarasimhaSLM
        slm = NarasimhaSLM(adapter_id=a.adapter) if a.adapter else NarasimhaSLM()
        print(f"[SLM] {slm.source}")
    if a.deberta:
        from src.slm import DebertaBaseline
        deb = DebertaBaseline(HF_DEBERTA_BASELINE)
    llm = ExternalLLM() if a.llm else None
    if llm is not None and not llm.enabled:
        print("[LLM] Sin OPENROUTER_API_KEY: la capa externa queda desactivada (modo local)."); llm = None

    rec = []
    for i, text in enumerate(df["prompt"]):
        r, t_r = timed(rules.scan, text)
        h, t_h = timed(heur.score, text)
        row = {"rules_fired": r.fired, "rules_matches": ";".join(r.matches), "rules_ms": t_r,
               "H": h.score, "heuristics": ";".join(h.indicators), "heuristic_ms": t_h}
        if slm:
            s = slm.classify(text); row.update(slm_label=s["label"], slm_raw=s["raw"], slm_ms=s["latency_ms"])
        if deb:
            d = deb.classify(text); row.update(deberta_flag=d["flagged"], deberta_ms=d["latency_ms"])
        if llm:
            g = llm.analyze(text, h.indicators)
            row.update(llm_ok=g.get("ok", False), llm_flag=g.get("flagged", False),
                       llm_conf=g.get("confidence"), llm_ms=g.get("latency_ms"))
        rec.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(df)} prompts procesados")
    pr = pd.concat([df.reset_index(drop=True), pd.DataFrame(rec)], axis=1)

    # ---- Configuraciones independientes (Tabla 5) ----
    configs = {"Rules/Regex (local)": (pr["rules_fired"].to_numpy(), pr["rules_ms"].mean())}
    if slm:
        configs["SLM Narasimha-0.6B"] = ((pr["slm_label"] != "benign").to_numpy(), pr["slm_ms"].mean())
    if deb:
        configs["DeBERTa-v3 PI-v2"] = (pr["deberta_flag"].to_numpy(bool), pr["deberta_ms"].mean())
    if llm:
        configs["Gemini 3 Flash"] = (pr["llm_flag"].to_numpy(bool), pr["llm_ms"].mean())

    # ---- Arquitectura con enrutamiento (sección 5.4) ----
    slm_attack = pr["slm_label"].isin(["sensitive", "jailbreak"]).to_numpy() if slm else np.zeros(len(pr), bool)
    retained = pr["rules_fired"].to_numpy() | slm_attack
    if llm:
        ext_flag, mode = pr["llm_flag"].to_numpy(bool), "LLM externo"
    else:
        ext_flag, mode = (pr["H"] >= 0.5).to_numpy(), "modo local (heurística H >= 0.5 en lugar del LLM externo)"
    routed = retained | (~retained & ext_flag)
    pr["retained_local"], pr["routed_pred"] = retained, routed
    configs["Routed hybrid"] = (routed, None)

    rows = []
    for name, (pred, lat) in configs.items():
        m = binary_metrics(y_true, pred)
        lo, hi = bootstrap_accuracy_ci(y_true, pred, BOOTSTRAP_RESAMPLES, SEED)
        m.update(acc_ci=f"[{lo:.1%}, {hi:.1%}]", latency_ms=lat, **per_category(pr["label"], pred, pr["subtype"]))
        ref = PAPER_RESULTS.get(name, {})
        m.update({f"paper_{k}": v for k, v in ref.items()})
        rows.append({"config": name, **m})
    met = pd.DataFrame(rows)

    sens = (pr["label"] == "sensitive").to_numpy()
    k_sens = int((retained & sens).sum()); n_sens = int(sens.sum())
    lat_local = pr["rules_ms"].mean() + (pr["slm_ms"].mean() if slm else 0)
    lat_ext = lat_local + (pr["llm_ms"].mean() if llm else 0)
    share_local = retained.mean()
    routing = {
        "fuente_datos": source, "slm": slm.source if slm else "no usado", "modo_externo": mode, "perfil_reglas": a.rules_profile,
        "retenidos_local": f"{int(retained.sum())}/{len(pr)}", "enviados_externo": f"{int((~retained).sum())}/{len(pr)}",
        "filtro_por_SLM": int((slm_attack & ~pr['rules_fired'].to_numpy()).sum()) if slm else 0,
        "filtro_por_reglas": int(pr["rules_fired"].sum()),
        "sensibles_retenidos": f"{k_sens}/{n_sens}", "sensibles_retenidos_IC95_wilson": [round(x, 3) for x in wilson_ci(k_sens, n_sens)],
        "sensibles_enviados_fuera": [p for p, s, r_ in zip(pr["prompt"], sens, retained) if s and not r_],
        "latencia_camino_local_ms": round(lat_local, 1), "latencia_camino_externo_ms": round(lat_ext, 1),
        "latencia_media_ponderada_ms": round(share_local * lat_local + (1 - share_local) * lat_ext, 1),
        "articulo": PAPER_ROUTING,
        "entorno": {"python": platform.python_version(), "cpu": platform.processor() or platform.machine(), "so": platform.platform()},
    }

    pr.to_csv(out / "predicciones.csv", index=False)
    met.to_csv(out / "metricas.csv", index=False)
    (out / "enrutamiento.json").write_text(json.dumps(routing, indent=2, ensure_ascii=False), encoding="utf-8")

    show = met[["config", "accuracy", "acc_ci", "precision", "recall", "f1", "fpr", "latency_ms",
                "benign_specificity", "jailbreak_recall", "sensitive_recall"]].copy()
    print("\n=== Métricas (réplica) ===")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Enrutamiento ===")
    print(json.dumps({k: v for k, v in routing.items() if k != "entorno"}, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados en {out.resolve()}")


if __name__ == "__main__":
    main()
