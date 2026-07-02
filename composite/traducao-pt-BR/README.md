# Tradução PT-BR — *Planets in Composite* (Robert Hand)

Projeto isolado. Meta: traduzir **na íntegra** para português do Brasil o livro
`../composite_planets_restructured.md` (~125 mil palavras), com **tradução feita
pelo modelo Haiku em várias sessões**, mantendo **consistência e fidelidade**
entre as sessões.

A estratégia para garantir qualidade apesar de muitas sessões independentes:
1. **Glossário travado** (`GLOSSARIO.md`) — mesma terminologia sempre.
2. **Guia de estilo** (`GUIA-DE-ESTILO.md`) — mesma voz, tom e regras sempre.
3. **Blocos determinísticos** (`MANIFEST.csv`) — o livro já foi fatiado em 69
   blocos nas fronteiras de títulos; cada sessão traduz um bloco, sem
   sobreposição nem lacuna.
4. **Rastreamento de progresso** — o MANIFEST registra o que está feito, por
   quem, e de quais linhas-fonte; a montagem final é só concatenar.

## Estrutura de pastas
```
traducao-pt-BR/
├── CLAUDE.md               ← INSTRUÇÕES CENTRAIS (auto-carregadas): papel + regras
├── README.md               ← este arquivo (protocolo detalhado)
├── GLOSSARIO.md            ← terminologia OBRIGATÓRIA e travada
├── GUIA-DE-ESTILO.md       ← voz, tom, OCR, markdown, checagem final
├── GLOSSARIO-CANDIDATOS.md ← fila de termos novos a oficializar
├── MANIFEST.csv            ← 69 blocos: linhas-fonte, status, responsável
├── PROGRESSO.md            ← painel legível de andamento
├── ferramentas.py          ← utilitário (extrair bloco, status, montar)
├── blocos/                 ← saídas: 001.md … 069.md
└── RESULTADO.md            ← gerado por `ferramentas.py montar` (final)
```

## PROTOCOLO DE UMA SESSÃO DE TRADUÇÃO (Haiku) — passo a passo

> Cada sessão traduz **um** bloco (ou alguns, se sobrar orçamento). É curta,
> autocontida e reprodutível. Faça exatamente isto:

0. **Sincronize primeiro:** `git pull` na branch de trabalho (evita retraduzir
   blocos já feitos e conflitos no MANIFEST).
1. **Carregue o contexto fixo** (sempre, toda sessão):
   - `CLAUDE.md` (auto-carregado — papel e regras invioláveis);
   - leia `GLOSSARIO.md` inteiro;
   - leia `GUIA-DE-ESTILO.md` inteiro.
2. **Pegue o próximo bloco pendente:**
   ```
   python3 ferramentas.py proximo
   ```
   (mostra o `block_id`, as linhas-fonte e o arquivo de saída).
3. **Extraia o trecho-fonte EXATO** desse bloco:
   ```
   python3 ferramentas.py extrair NNN
   ```
   Traduza **somente** esse trecho.
4. **Traduza** aplicando glossário + guia de estilo. Preserve títulos,
   `<!-- image -->`, tabelas e a estrutura de parágrafos.
5. **Salve** a tradução em `blocos/NNN.md` (só o texto traduzido).
6. **Verifique a paridade estrutural** (guardião contra omissão):
   ```
   python3 ferramentas.py verificar NNN
   ```
   Só siga com **APROVADO**. Depois rode a checagem final do GUIA-DE-ESTILO §7.
7. **Registre termos novos** (se houver) em `GLOSSARIO-CANDIDATOS.md`.
8. **Marque o bloco como concluído:**
   ```
   python3 ferramentas.py concluir NNN <identificador-da-sessao>
   ```
9. **Commit + push** com mensagem clara, ex.:
   `traducao: bloco NNN (<titulo>) PT-BR`.

> **Sessões em paralelo?** Antes do passo 3, reserve o bloco e publique a
> reserva: `python3 ferramentas.py reservar NNN <sessao>` + commit/push. Se o
> push for rejeitado, `git pull --rebase` e tente de novo; se o bloco já estiver
> tomado, pegue o próximo pendente.

## Regras de ouro entre sessões
- **Uma sessão nunca reescreve blocos já `done`** — evita divergência de estilo.
  Correções em bloco pronto só quando explicitamente pedido.
- **Não edite `GLOSSARIO.md` no meio de um bloco.** Sugira em CANDIDATOS; a
  oficialização é uma tarefa própria de curadoria.
- **Sempre trave o vocabulário no glossário**, mesmo que outra palavra pareça
  boa: consistência > preferência pessoal.

## Montar o resultado final
Quando (ou à medida que) blocos ficam prontos em ordem:
```
python3 ferramentas.py montar     # gera RESULTADO.md
python3 ferramentas.py status     # % concluído
```

## Curadoria periódica do glossário (recomendado a cada ~10 blocos)
Uma sessão dedicada revisa `GLOSSARIO-CANDIDATOS.md`, promove os termos bons
para `GLOSSARIO.md` e esvazia a fila. Isso mantém o vocabulário convergindo em
vez de divergir.
