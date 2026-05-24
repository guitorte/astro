"""LLM-based classification of a TikTok video against tiktok/SCHEMA.md.

Sends transcript + caption + metadata to Claude. Returns a dict whose
keys map to qualitative columns of 03_videos.csv. The system prompt
mirrors .claude/agents/astro-content-coder.md but is duplicated here
because that file is read by the Claude Code Agent tool, not by this
pipeline script.

CALIBRATION REQUIRED: before running on all 200 videos, manually code 20
videos, run them through this classifier, and compare. If hook_type or
sub_niche agreement is below 80%, tighten the prompt before scaling.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic

from . import config

SYSTEM_PROMPT = """Você codifica vídeos de TikTok de astrologia em PT-BR contra o schema definido em tiktok/SCHEMA.md. Aplique o schema de forma CONSISTENTE entre vídeos — drift de codificação é o que arruína a análise.

Recebe: transcrição PT-BR + legenda + metadados (views, likes, duração).
Retorna: JSON com EXATAMENTE estas chaves, sem prosa adicional:

{
  "hook_type": "question|hot_take|claim|story|list|demo|pattern_interrupt",
  "first_15s_topic": "<descrição curta em PT-BR do que é estabelecido nos primeiros 15s>",
  "format": "talking_head|text_only|b_roll|mixed",
  "sub_niche": "western_general|vedic|hellenistic|signs|ascendant|synastry|transits|houses|mundane|astro_psych|mixed",
  "specificity": "generic|specific|hyper_specific",
  "cta_type": "none|follow|comment|save|link_bio|dm",
  "has_text_overlay": "yes|no|unknown",
  "has_voiceover": "yes|no|unknown",
  "notes": "<sentença curta justificando qualquer decisão limítrofe; vazio se não aplicável>"
}

Regras:
- hook_type: decidido pelo elemento dominante nos primeiros 1.5s. Se incerto, escolha um e mencione a alternativa em notes.
- format: 'mixed' só se nenhum formato domina ≥ 60% da duração.
- sub_niche: deve estar na lista. Se um novo sub-niche parece se encaixar, NÃO invente — coloque 'mixed' e flag em notes.
- specificity: 'generic' (todos os signos), 'specific' (uma dimensão), 'hyper_specific' (duas+ dimensões).
- has_text_overlay e has_voiceover: 'unknown' quando a transcrição sozinha não permite decidir.
- Se faltar informação para algum campo, marque 'unknown' (ou 'mixed' onde 'unknown' não está na enum) e descreva em notes.

Retorne APENAS o JSON. Sem markdown, sem explicação."""


@dataclass
class ClassifyInput:
    transcript: str
    caption: Optional[str]
    length_seconds: Optional[int]
    views: Optional[int]
    likes: Optional[int]


def classify(inp: ClassifyInput, model: str | None = None) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Classification requires the Claude API."
        )

    client = Anthropic()
    user_msg = (
        f"Transcrição PT-BR:\n{inp.transcript}\n\n"
        f"Legenda: {inp.caption or '(sem legenda)'}\n"
        f"Duração: {inp.length_seconds or '?'} segundos\n"
        f"Views: {inp.views or '?'}\n"
        f"Likes: {inp.likes or '?'}\n"
    )

    response = client.messages.create(
        model=model or config.CLASSIFIER_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)
