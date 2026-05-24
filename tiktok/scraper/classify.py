"""Local LLM classification of a TikTok video against tiktok/SCHEMA.md.

Uses Ollama (https://ollama.com) running locally — no API key, no
network cost. Default model: qwen2.5:7b (balanced quality/hardware).
Swap via the OLLAMA_MODEL env var.

Hardware notes:
- 7b   ~5 GB  — runs on CPU (slow) or any modern GPU
- 14b  ~10 GB — needs ≥ 12 GB VRAM or fast CPU
- 32b  ~20 GB — needs ≥ 24 GB VRAM

CALIBRATION REQUIRED: before running on all 200 videos, manually code 20
videos, run them through this classifier, and compare. With a 7b model,
expect 5-15% lower agreement with manual coding than a frontier API
model — adjust the calibration threshold accordingly (target 70-80%
agreement on hook_type and sub_niche instead of 80% strict).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import ollama

from . import config

SYSTEM_PROMPT = """Você codifica vídeos de TikTok de astrologia em PT-BR contra um schema fixo. Aplique o schema de forma CONSISTENTE entre vídeos — drift de codificação arruína a análise.

Recebe: transcrição PT-BR + legenda + metadados (views, likes, duração).
Retorna: JSON estritamente conforme o schema fornecido, sem prosa adicional.

Regras de codificação:
- hook_type: elemento dominante nos primeiros 1.5s. Se incerto, escolha um e mencione a alternativa em notes.
- format: 'mixed' só se nenhum formato domina ≥ 60% da duração.
- sub_niche: deve estar na enum. Se um sub-niche novo parece encaixar, use 'mixed' e flag em notes.
- specificity: 'generic' (todos os signos), 'specific' (uma dimensão), 'hyper_specific' (duas+ dimensões).
- has_text_overlay e has_voiceover: 'unknown' quando a transcrição sozinha não permite decidir.
- Se faltar informação para algum campo categórico, escolha o valor mais conservador e explique em notes."""


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hook_type": {
            "type": "string",
            "enum": ["question", "hot_take", "claim", "story", "list", "demo", "pattern_interrupt"],
        },
        "first_15s_topic": {"type": "string"},
        "format": {
            "type": "string",
            "enum": ["talking_head", "text_only", "b_roll", "mixed"],
        },
        "sub_niche": {
            "type": "string",
            "enum": [
                "western_general", "vedic", "hellenistic", "signs", "ascendant",
                "synastry", "transits", "houses", "mundane", "astro_psych", "mixed",
            ],
        },
        "specificity": {
            "type": "string",
            "enum": ["generic", "specific", "hyper_specific"],
        },
        "cta_type": {
            "type": "string",
            "enum": ["none", "follow", "comment", "save", "link_bio", "dm"],
        },
        "has_text_overlay": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "has_voiceover": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "notes": {"type": "string"},
    },
    "required": [
        "hook_type", "first_15s_topic", "format", "sub_niche",
        "specificity", "cta_type", "has_text_overlay", "has_voiceover", "notes",
    ],
}


@dataclass
class ClassifyInput:
    transcript: str
    caption: Optional[str]
    length_seconds: Optional[int]
    views: Optional[int]
    likes: Optional[int]


def classify(inp: ClassifyInput, model: str | None = None) -> dict:
    user_msg = (
        f"Transcrição PT-BR:\n{inp.transcript}\n\n"
        f"Legenda: {inp.caption or '(sem legenda)'}\n"
        f"Duração: {inp.length_seconds or '?'} segundos\n"
        f"Views: {inp.views or '?'}\n"
        f"Likes: {inp.likes or '?'}\n"
    )

    client = ollama.Client(host=config.OLLAMA_HOST)
    try:
        response = client.chat(
            model=model or config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format=RESPONSE_SCHEMA,
            options={"temperature": 0.1, "num_ctx": 8192},
        )
    except ConnectionError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_HOST}. Is `ollama serve` running? "
            f"Original error: {e}"
        ) from e

    text = response["message"]["content"].strip()
    return json.loads(text)
