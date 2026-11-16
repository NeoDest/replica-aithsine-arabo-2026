"""Motor heurístico (sección 4.4, p. 8).

Indicadores descritos por los autores: lenguaje para anular instrucciones, role-play,
meta-prompting, prefijos que cambian el contexto ("for research", "in a hypothetical")
y manipulación en varias etapas. Devuelve un puntaje de confianza H en [0, 1], no una
decisión binaria. Los pesos son una RECONSTRUCCIÓN: el artículo no los publica.
"""
import re
from dataclasses import dataclass, field

INDICATORS = {
    "instruction_override": (0.45, r"\b(?:ignore|disregard|override|forget)\b[^.\n]{0,40}\b(?:instructions|rules|guidelines|policy|policies)\b"),
    "role_play": (0.30, r"\b(?:pretend|act as|you are now|role[- ]?play|imagine you are)\b"),
    "meta_prompting": (0.30, r"\b(?:system prompt|your (?:instructions|rules|guidelines)|hidden prompt|initial prompt)\b"),
    "context_shift": (0.25, r"\b(?:for research(?: purposes)?|hypothetical(?:ly)?|in a fictional|for a novel|for educational purposes|just curious)\b"),
    "bypass_language": (0.30, r"\b(?:no restrictions|without (?:any )?(?:filters|limits|restrictions)|unfiltered|uncensored|jailbreak)\b"),
    "multi_stage": (0.20, r"\b(?:step 1|first,? (?:forget|ignore)|then,? (?:tell|reveal|answer)|from now on)\b"),
}


@dataclass
class HeuristicResult:
    score: float
    indicators: list = field(default_factory=list)


class HeuristicEngine:
    def __init__(self):
        self.rx = {k: (w, re.compile(p, re.I)) for k, (w, p) in INDICATORS.items()}

    def score(self, text: str) -> HeuristicResult:
        hits = [k for k, (w, r) in self.rx.items() if r.search(text)]
        h = min(1.0, sum(self.rx[k][0] for k in hits))
        return HeuristicResult(score=round(h, 3), indicators=hits)
