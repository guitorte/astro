# tiktok/scraper — pipeline de coleta e classificação

Pipeline em Python que descobre, baixa, transcreve e classifica vídeos de TikTok em PT-BR contra o schema em `../SCHEMA.md`. Saídas alimentam diretamente `../02_accounts.csv` e `../03_videos.csv`.

## Arquitetura

```
discover   →  data/discovery.json
metadata   →  03_videos.csv (colunas quantitativas)
sample     →  marca best / worst / random_1..3 em 03_videos.csv
download   →  data/videos/*.mp4
transcribe →  data/transcripts/*.txt
classify   →  03_videos.csv (colunas qualitativas via LLM local)
```

`pilot` roda tudo end-to-end em N contas para validar antes de escalar.

## Dependências de sistema

- `ffmpeg` (necessário para yt-dlp processar áudio/vídeo)
- Python 3.11+
- `ollama` — servidor de LLM local. Instalar de https://ollama.com (Linux: `curl -fsSL https://ollama.com/install.sh | sh`)
- Modelos do Whisper são baixados na primeira execução (~1.5 GB para `medium`, ~3 GB para `large-v3`)
- Modelo do Ollama é baixado uma vez: `ollama pull qwen2.5:7b` (~5 GB)

Instalar Python deps:

```
pip install -r requirements.txt
```

Iniciar o servidor Ollama (deixar rodando em background):

```
ollama serve
```

Variáveis de ambiente:

```
export WHISPER_MODEL=medium            # ou large-v3 se tiver GPU
export WHISPER_DEVICE=auto             # auto / cuda / cpu
export TIKTOK_REQUEST_DELAY=3.0        # segundos entre requests
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=qwen2.5:7b         # ou qwen2.5:14b se tiver hardware
```

## Escolha de modelo Ollama

Pra classificação contra schema fixo em PT-BR, vale a pena experimentar:

| Modelo | Tamanho | RAM/VRAM | Qualidade esperada vs. frontier |
|---|---|---|---|
| `qwen2.5:7b` | ~5 GB | ≥ 8 GB | 5-15% menos concordância — default razoável |
| `qwen2.5:14b` | ~10 GB | ≥ 12 GB VRAM ou CPU rápida | 3-8% menos concordância |
| `qwen2.5:32b` | ~20 GB | ≥ 24 GB VRAM | comparável |
| `llama3.1:8b` | ~5 GB | ≥ 8 GB | alternativa com PT-BR ligeiramente mais natural |
| `gemma2:9b` | ~6 GB | ≥ 10 GB | alternativa multilíngue forte |

Confiança: moderada. Os números acima são estimativas baseadas em benchmarks gerais — a degradação real específica para este task (classificação contra schema fixo, transcrições PT-BR) tem que ser medida na fase de calibração.

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
4. **Threshold realista com Ollama 7B**: 70-75% de concordância. Com 14B: 75-80%. Com 32B+: 80%+. Modelos open-source locais drifta mais em casos limítrofes que frontier APIs.
5. Se concordância abaixo do threshold do seu modelo, ajustar o `SYSTEM_PROMPT` em `classify.py` ou trocar pra modelo maior antes de continuar.

## Limitações conhecidas e níveis de confiança

- **Discovery via hashtag (confiança: moderada-baixa).** A TikTok crescentemente exige login para listagens de hashtag. Se `discover` retorna zero entradas, passe uma lista manual de handles para `metadata --handles`.
- **Campo `saves` (confiança: moderada).** `yt-dlp` expõe `save_count` / `collect_count` inconsistentemente entre versões. O pipeline tenta múltiplos campos do JSON; quando ausente, deixa em branco. Validar no piloto contra a contagem visível no ícone de bookmark da página pública.
- **Sem coleta de comentários (confiança: alta, por design).** Coleta de comentários sem login retorna apenas as top-N visíveis (não há scroll infinito). Para Q5/Q6 do plano de pesquisa, use o agente `astro-comment-analyzer` colando dumps manuais de comentários a partir da página pública.
- **Detecção de scraping (confiança: moderada).** TikTok pode rate-limitar IPs com padrão de scraping. Mitigação built-in: `TIKTOK_REQUEST_DELAY` (default 3s). Se receber 429/403 sistemático, aumentar para 10-15s ou rodar em chunks.
- **Acurácia da classificação por LLM local (confiança: alta nos mecanismos, baixa-moderada na qualidade).** Depende inteiramente da calibração descrita acima E do modelo escolhido. Qwen2.5:7b é um trade-off, não uma solução ótima — se a calibração reprovar repetidamente, suba pra 14b ou 32b antes de descartar o pipeline.
- **Idiomas / regiões (confiança: alta para PT-BR padrão, moderada para sotaques fortes).** Whisper `medium` lida bem com PT-BR neutro/sudeste. Para sotaques NE pesados, usar `large-v3`.

## Quando o pipeline NÃO substitui codificação manual

- Validação dos 20 vídeos iniciais para calibração.
- Spot-checking de ~10% das classificações antes de mover para `04_findings.md`.
- Casos onde o LLM marca `unknown` em vários campos — vídeo precisa de revisão humana.
