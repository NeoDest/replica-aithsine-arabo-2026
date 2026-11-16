"""Middleware: controlador, enrutador de sensibilidad y motor de veredicto.

Flujo (Fig. 2, p. 10): prompt -> tokenización y patrones -> heurística -> SLM ->
¿sensible? sí: solo local / no: LLM externo -> motor de veredicto -> acción.

Regla del filtro usada en la evaluación del artículo (sección 5.4, pp. 15-16):
un prompt se retiene en local si el módulo de patrones se activa O si el SLM lo
clasifica como ataque (sensitive o jailbreak); si no, se reenvía al LLM externo.
"""
import time
from dataclasses import dataclass, field

from .config import ACTIONS, CRITICAL_THRESHOLD, W_HEURISTIC, W_LLM
from .heuristics import HeuristicEngine
from .rules import RuleEngine


@dataclass
class Decision:
    retained_local: bool
    flagged: bool
    tier: str
    action: str
    confidence: float
    path: str
    detail: dict = field(default_factory=dict)


class Middleware:
    def __init__(self, slm=None, external=None, rules_profile: str = "full", local_threshold: float = 0.5):
        self.rules = RuleEngine(profile=rules_profile)
        self.heur = HeuristicEngine()
        self.slm = slm                    # NarasimhaSLM o None
        self.external = external          # ExternalLLM o None
        self.local_threshold = local_threshold

    def process(self, text: str) -> Decision:
        lat = {}
        t = time.perf_counter(); r = self.rules.scan(text); lat["rules_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); h = self.heur.score(text); lat["heuristic_ms"] = (time.perf_counter() - t) * 1000
        slm_out = self.slm.classify(text) if self.slm else {"label": None, "latency_ms": 0.0}
        lat["slm_ms"] = slm_out["latency_ms"]

        # Enrutador de sensibilidad: decisión binaria con señales locales (p. 8)
        slm_attack = slm_out["label"] in ("sensitive", "jailbreak")
        retained = r.fired or slm_attack

        llm = {"ok": False}
        if not retained and self.external is not None and self.external.enabled:
            llm = self.external.analyze(text, h.indicators)
            lat["llm_ms"] = llm.get("latency_ms", 0.0)

        # Ecuación (1), p. 12: C = 0.40 H + 0.60 L solo si se invocó el LLM externo
        C = W_HEURISTIC * h.score + W_LLM * llm["confidence"] if llm.get("ok") else h.score

        if retained:
            flagged, path = True, "local"
        elif llm.get("ok"):
            flagged, path = llm["flagged"], "external"
        else:                              # modo local: sin LLM externo disponible
            flagged, path = h.score >= self.local_threshold, "external_unavailable"

        # Cascada de niveles (sección 4.8, p. 12). RECONSTRUCCIÓN: el artículo da el orden
        # (contenido dañino, intento de evasión, firma de ataque) y el umbral C >= 0.70.
        harmful = r.harmful or llm.get("harmful", False)
        if harmful and C >= CRITICAL_THRESHOLD:
            tier = "critical_harm"
        elif harmful or r.attack_signature or slm_out["label"] == "jailbreak" or (path == "external" and flagged):
            tier = "security_attack"
        elif flagged:
            tier = "concerning"
        else:
            tier = "benign"

        lat["path_ms"] = sum(v for k, v in lat.items() if k.endswith("_ms"))
        return Decision(retained_local=retained, flagged=flagged, tier=tier, action=ACTIONS[tier],
                        confidence=round(C, 3), path=path,
                        detail={"rules": r.matches, "heuristics": h.indicators, "H": h.score,
                                "slm_label": slm_out["label"], "llm": llm, "latency": lat})
