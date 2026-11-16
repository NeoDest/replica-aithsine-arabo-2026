"""Clasificador SLM local (secciones 4.4 y 4.7, pp. 8-11).

- Modelo base: Narasimha-0.6B (sarthakrastogi/narasimha-b-0.6b), decoder de la familia Qwen3.
- Adaptador LoRA publicado por los autores: v1adam/slm-threat-detection-v1.
- Clasificación generativa con la plantilla de la p. 11; el primer token generado
  tras "Classification:" se interpreta como la etiqueta.
- Inferencia solo en CPU (sección 4.10, p. 13).
"""
import time

from .config import CLASSES, HF_SLM_ADAPTER, HF_SLM_BASE, SLM_TEMPLATE


class NarasimhaSLM:
    def __init__(self, base_id: str = HF_SLM_BASE, adapter_id: str = HF_SLM_ADAPTER, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        try:
            from peft import PeftConfig, PeftModel
            cfg = PeftConfig.from_pretrained(adapter_id)
            base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path or base_id, torch_dtype=torch.float32)
            self.model = PeftModel.from_pretrained(base, adapter_id)
            self.tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path or base_id)
            self.source = f"{cfg.base_model_name_or_path or base_id} + LoRA {adapter_id}"
        except Exception as e:  # si el repositorio contiene el modelo completo y no un adaptador
            print(f"[SLM] No se pudo cargar como adaptador LoRA ({e}). Se intenta como modelo completo.")
            self.model = AutoModelForCausalLM.from_pretrained(adapter_id, torch_dtype=torch.float32)
            self.tok = AutoTokenizer.from_pretrained(adapter_id)
            self.source = adapter_id
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
