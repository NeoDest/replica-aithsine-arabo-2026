"""Clasificador SLM local (secciones 4.4 y 4.7, pp. 8-11).

- Modelo base: Narasimha-0.6B (sarthakrastogi/narasimha-b-0.6b), decoder de la familia Qwen3.
- Adaptador LoRA publicado por los autores: v1adam/slm-threat-detection-v1.
  La Tabla 3 (p. 11) dice que sus valores salen del config.json y el adapter_config.json
  del checkpoint publicado. En el repositorio, esos archivos pueden estar en una subcarpeta
  (p. ej., checkpoint-175: 175 pasos de optimización); por eso se buscan automáticamente.
- Clasificación generativa con la plantilla de la p. 11; el primer token generado
  tras "Classification:" se interpreta como la etiqueta.
- Inferencia solo en CPU (sección 4.10, p. 13).
"""
import os
import re
import time

from .config import CLASSES, HF_SLM_ADAPTER, HF_SLM_BASE, SLM_TEMPLATE


def _pick_subfolder(files, filename):
    """Devuelve la subcarpeta que contiene `filename`. Si hay varios checkpoints, el último
    (el artículo publica el de la última época, p. 11)."""
    hits = [f for f in files if f == filename or f.endswith("/" + filename)]
    if not hits:
        return None
    def step(path):
        m = re.search(r"checkpoint-(\d+)", path)
        return int(m.group(1)) if m else 10 ** 9      # carpeta sin número = versión final
    best = max(hits, key=lambda p: (step(p), -p.count("/")))
    return os.path.dirname(best)                      # "" si está en la raíz


def _valid_hub_id(name):
    return bool(name) and re.fullmatch(r"[\w.-]+/[\w.-]+", name) is not None


def _safe_extract(zip_path, dest):
    """Descomprime evitando rutas fuera de la carpeta de destino."""
    import zipfile
    dest = os.path.abspath(dest)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            target = os.path.abspath(os.path.join(dest, m))
            if not target.startswith(dest + os.sep) and target != dest:
                raise ValueError(f"Ruta no permitida en el zip: {m}")
        z.extractall(dest)


def _local_files(root):
    return [os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
            for d, _, fs in os.walk(root) for f in fs if "__MACOSX" not in d]


class NarasimhaSLM:
    def __init__(self, base_id: str = HF_SLM_BASE, adapter_id: str = HF_SLM_ADAPTER, device: str = "cpu"):
        import torch
        from huggingface_hub import hf_hub_download, list_repo_files
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        files = list_repo_files(adapter_id)
        print(f"[SLM] Archivos en {adapter_id}: {files}")

        # Los autores publicaron el checkpoint comprimido (slm.zip): se descarga y se descomprime.
        root, where = adapter_id, adapter_id
        zips = [f for f in files if f.lower().endswith(".zip")]
        if not any(f.endswith(("adapter_config.json", "config.json")) for f in files) and zips:
            zpath = hf_hub_download(adapter_id, zips[0])
            root = os.path.join(os.path.expanduser("~"), ".cache", "slm_extract", adapter_id.replace("/", "__"))
            if not os.path.isdir(root) or not os.listdir(root):
                os.makedirs(root, exist_ok=True)
                _safe_extract(zpath, root)
            files = _local_files(root)
            where = f"{adapter_id}/{zips[0]}"
            print(f"[SLM] Contenido de {zips[0]}:")
            for f in files:
                print("   ", f)

        def load_path(sub):
            return os.path.join(root, sub) if os.path.isdir(root) else root, \
                   ({} if os.path.isdir(root) or not sub else {"subfolder": sub})

        adapter_sub = _pick_subfolder(files, "adapter_config.json")
        if adapter_sub is not None:
            from peft import PeftConfig, PeftModel
            path, kw = load_path(adapter_sub)
            cfg = PeftConfig.from_pretrained(path, **kw)
            base_name = cfg.base_model_name_or_path if _valid_hub_id(cfg.base_model_name_or_path) else base_id
            print(f"[SLM] Modelo base: {base_name} (en adapter_config.json: {cfg.base_model_name_or_path})")
            base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32)
            self.model = PeftModel.from_pretrained(base, path, **kw)
            tok_sub = _pick_subfolder(files, "tokenizer_config.json")
            if tok_sub is None:            # el adaptador no trae tokenizer: se usa el del modelo base
                self.tok = AutoTokenizer.from_pretrained(base_name)
            else:
                tpath, tkw = load_path(tok_sub)
                self.tok = AutoTokenizer.from_pretrained(tpath, **tkw)
            self.source = f"{base_name} + LoRA {where} ({adapter_sub or 'raíz'})"
        else:
            full_sub = _pick_subfolder(files, "config.json")
            if full_sub is None:
                raise RuntimeError(f"No hay adapter_config.json ni config.json en {where}. Archivos: {files}")
            path, kw = load_path(full_sub)
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, **kw)
            self.tok = AutoTokenizer.from_pretrained(path, **kw)
            self.source = f"{where} ({full_sub or 'raíz'}, modelo completo)"
        self.model.to(device).eval()

    @staticmethod
    def parse(generated: str) -> str:
        word = generated.strip().lower().lstrip(":.\"' ")
        for c in CLASSES:
            if word.startswith(c[:4]):      # "ben", "sens", "jail"
                return c
        return "benign"                     # salida no interpretable -> se trata como benigno

    def classify(self, text: str) -> dict:
        prompt = SLM_TEMPLATE.format(prompt=text)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        ms = (time.perf_counter() - t0) * 1000
        gen = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"label": self.parse(gen), "raw": gen, "latency_ms": ms}


class DebertaBaseline:
    """Línea base local de la Tabla 5 (p. 14): encoder DeBERTa-v3 para prompt injection."""

    def __init__(self, model_id: str):
        from transformers import pipeline
        self.clf = pipeline("text-classification", model=model_id, device=-1, truncation=True, max_length=512)

    def classify(self, text: str) -> dict:
        t0 = time.perf_counter()
        r = self.clf(text)[0]
        ms = (time.perf_counter() - t0) * 1000
        return {"flagged": r["label"].upper() != "SAFE", "label": r["label"], "latency_ms": ms}
