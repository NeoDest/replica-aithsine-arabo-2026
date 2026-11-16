"""Módulo de tokenización y coincidencia de patrones (sección 4.4, p. 8).

El artículo describe los tipos de patrones, pero no publica la lista exacta de expresiones:
  - PII: correo, teléfono, identificadores nacionales, números de cuenta financiera.
  - Material de autenticación: API keys, tokens, cadenas de conexión.
  - Firmas sintácticas conocidas de jailbreak.
  - Validación de identificadores financieros con checksum de Luhn.
  - Umbral de entropía para cadenas de alta entropía (tokens, claves).
Las expresiones de abajo son una RECONSTRUCCIÓN a partir de esa descripción.
Nota (sección 6.3, p. 18): la versión de los autores no detectó AWS keys, webhooks de Slack,
encabezados de clave OpenSSH ni cadenas de MongoDB. Para acercarse a su motor, use
`profile="paper"`, que desactiva esos patrones.
"""
import math
import re
from dataclasses import dataclass, field

PII = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)\d{3,4}[\s-]?\d{3,4}\b",
    "us_ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "uk_nino": r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
    "passport": r"\bpassport\b[^\n]{0,30}?\b[A-Z]\d{7,8}\b",
    "iban": r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b",
}
CARD = r"\b(?:\d[ -]?){13,19}\b"                     # candidato; se valida con Luhn

AUTH_COMMON = {
    "password_in_url": r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@]+@[^\s]+",   # nombre citado en p. 18
    "bearer_token": r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}=*",
    "jwt": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    "generic_api_key": r"\b(?:api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}",
    "openai_key": r"\bsk-[A-Za-z0-9]{20,}",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    "password_assign": r"(?:\b|_)password\s*[:=]\s*\S{6,}",
}
AUTH_EXTRA = {   # formatos que el motor de los autores NO detectó (p. 18)
    "aws_access_key": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "slack_webhook": r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+",
    "private_key_header": r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----",
    "mongodb_uri": r"\bmongodb(?:\+srv)?://[^\s]+",
}
JAILBREAK_SIGNATURES = {
    "ignore_instructions": r"\b(?:ignore|disregard|forget)\b[^.\n]{0,30}\b(?:previous|prior|above|all)\b[^.\n]{0,20}\b(?:instructions|rules|prompts?)\b",
    "dan": r"\b(?:DAN|do anything now)\b",
    "developer_mode": r"\bdeveloper mode\b",
    "system_prompt_leak": r"\b(?:reveal|print|show|repeat)\b[^.\n]{0,25}\b(?:system prompt|hidden instructions)\b",
    "no_restrictions": r"\b(?:without|no)\s+(?:any\s+)?(?:restrictions|filters|limitations|guidelines)\b",
}
HARMFUL_CONTENT = {   # categorías del Apéndice A (p. 22): suicide, violence, abuse, self-harm
    "self_harm": r"\b(?:kill myself|suicide|self[- ]harm|end my life)\b",
    "violence": r"\b(?:build a bomb|make a weapon|hurt (?:someone|people)|kill (?:someone|him|her|them))\b",
    "abuse": r"\b(?:abuse a child|stalk (?:her|him|someone))\b",
}


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


@dataclass
class RuleResult:
    fired: bool
    sensitive: bool
    attack_signature: bool
    harmful: bool
    matches: list = field(default_factory=list)


class RuleEngine:
    def __init__(self, profile: str = "full", entropy_threshold: float = 4.0, min_token_len: int = 24):
        self.auth = dict(AUTH_COMMON)
        if profile == "full":
            self.auth.update(AUTH_EXTRA)
        self.pii = {k: re.compile(v, re.I if k == "passport" else 0) for k, v in PII.items()}
        self.auth_re = {k: re.compile(v, re.I) for k, v in self.auth.items()}
        self.jb_re = {k: re.compile(v, re.I) for k, v in JAILBREAK_SIGNATURES.items()}
        self.harm_re = {k: re.compile(v, re.I) for k, v in HARMFUL_CONTENT.items()}
        self.card_re = re.compile(CARD)
        self.entropy_threshold = entropy_threshold
        self.min_token_len = min_token_len

    def scan(self, text: str) -> RuleResult:
        m = []
        m += [f"pii:{k}" for k, r in self.pii.items() if r.search(text)]
        m += [f"auth:{k}" for k, r in self.auth_re.items() if r.search(text)]
        m += ["pii:card_luhn" for c in self.card_re.findall(text) if luhn_ok(c)][:1]
        for tok in re.findall(r"[A-Za-z0-9+/_\-=]{%d,}" % self.min_token_len, text):
            if shannon_entropy(tok) >= self.entropy_threshold:
                m.append("auth:high_entropy")
                break
        jb = [f"jailbreak:{k}" for k, r in self.jb_re.items() if r.search(text)]
        harm = [f"harm:{k}" for k, r in self.harm_re.items() if r.search(text)]
        sensitive = bool(m)
        return RuleResult(fired=bool(m or jb or harm), sensitive=sensitive,
                          attack_signature=bool(jb), harmful=bool(harm), matches=m + jb + harm)
