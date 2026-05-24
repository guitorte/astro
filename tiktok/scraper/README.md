# tiktok/scraper — pipeline de coleta e classificação

Pipeline em Python que descobre, baixa, transcreve e classifica vídeos de TikTok em PT-BR contra o schema em `../SCHEMA.md`. Saídas alimentam diretamente `../02_accounts.csv` e `../03_videos.csv`.

## Arquitetura

```
discover   →  data/discovery.json
metadata   →  03_videos.csv (colunas quantitativas)
sample     →  marca best / worst / random_1..3 em 03_videos.csv
download   →  data/videos/*.mp4
transcribe →  data/transcripts/*.txt
classify   →  03_videos.csv (colunas qualitativas via Claude API)
```

`pilot` roda tudo end-to-end em N contas para validar antes de escalar.

## Dependências de sistema

- `ffmpeg` (necessário para yt-dlp processar áudio/vídeo)
- Python 3.11+
- Modelos do Whisper são baixados na primeira execução (~1.5 GB para `medium`, ~3 GB para `large-v3`)

Instalar Python deps:

```
pip install -r requirements.txt
```

Variáveis de ambiente:

```
export ANTHROPIC_API_KEY=...           # obrigatória para classify
export WHISPER_MODEL=medium            # ou large-v3 se tiver GPU
export WHISPER_DEVICE=auto             # auto / cuda / cpu
export TIKTOK_REQUEST_DELAY=3.0        # segundos entre requests
```

## Uso

Piloto (recomendado primeiro):

```
python -m tiktok.scraper.pipeline pilot \
  --hashtags astrologia mapaastral signos ascendente \
  --n 5 \
  --classify
```

Por etapa, em produção:

```
python -m tiktok.scraper.pipeline discover --hashtags astrologia mapaastral signos
python -m tiktok.scraper.pipeline metadata --handles @conta1 @conta2 @conta3
python -m tiktok.scraper.pipeline sample
python -m tiktok.scraper.pipeline download
python -m tiktok.scraper.pipeline transcribe
python -m tiktok.scraper.pipeline classify
```

## Calibração obrigatória antes de escalar

Antes de rodar `classify` nos 200 vídeos do dataset principal:

1. Codificar manualmente 20 vídeos representativos (mistura de tiers e sub-nichos).
2. Rodar `classify` nos mesmos 20.
3. Comparar concordância em `hook_type` e `sub_niche` — os campos mais subjetivos.
4. Se concordância < 80% em qualquer um dos dois, ajustar o `SYSTEM_PROMPT` em `classify.py` antes de continuar. A análise inteira herda o viés desse prompt; calibrar uma vez economiza recodificar duas vezes.

## Limitações conhecidas e níveis de confiança

- **Discovery via hashtag (confiança: moderada-baixa).** A TikTok crescentemente exige login para listagens de hashtag. Se `discover` retorna zero entradas, passe uma lista manual de handles para `metadata --handles`.
- **Campo `saves` (confiança: moderada).** `yt-dlp` expõe `save_count` / `collect_count` inconsistentemente entre versões. O pipeline tenta múltiplos campos do JSON; quando ausente, deixa em branco. Validar no piloto contra a contagem visível no ícone de bookmark da página pública.
- **Sem coleta de comentários (confiança: alta, por design).** Coleta de comentários sem login retorna apenas as top-N visíveis (não há scroll infinito). Para Q5/Q6 do plano de pesquisa, use o agente `astro-comment-analyzer` colando dumps manuais de comentários a partir da página pública.
- **Detecção de scraping (confiança: moderada).** TikTok pode rate-limitar IPs com padrão de scraping. Mitigação built-in: `TIKTOK_REQUEST_DELAY` (default 3s). Se receber 429/403 sistemático, aumentar para 10-15s ou rodar em chunks.
- **Acurácia da classificação por LLM (confiança: alta nos mecanismos, moderada na qualidade).** Depende inteiramente da calibração descrita acima.
- **Idiomas / regiões (confiança: alta para PT-BR padrão, moderada para sotaques fortes).** Whisper `medium` lida bem com PT-BR neutro/sudeste. Para sotaques NE pesados, usar `large-v3`.

## Quando o pipeline NÃO substitui codificação manual

- Validação dos 20 vídeos iniciais para calibração.
- Spot-checking de ~10% das classificações antes de mover para `04_findings.md`.
- Casos onde o LLM marca `unknown` em vários campos — vídeo precisa de revisão humana.
