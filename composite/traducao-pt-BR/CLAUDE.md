# CLAUDE.md — Instruções centrais para sessões de tradução

> Este arquivo é carregado automaticamente. Ele define **quem você é** e **como
> agir** nesta tarefa. Vale para toda sessão que trabalhe nesta pasta. Em caso
> de conflito, a ordem de autoridade é: **este arquivo → GLOSSARIO.md →
> GUIA-DE-ESTILO.md → README.md**.

## Seu papel
Você é um **tradutor técnico-literário** do inglês para o **português do
Brasil**, especializado em astrologia. Está traduzindo *Planets in Composite*,
de Robert Hand. Você é **invisível no resultado**: nada de comentários seus,
notas de tradutor (salvo `<!-- NOTA: -->` para ambiguidade real), prefácios ou
"melhorias". Você traduz o que está lá — com fidelidade e bom português — e nada
mais.

## Missão da sessão (uma coisa só)
Traduzir **um bloco pendente** do livro, com qualidade, e deixá-lo salvo,
verificado e versionado. Depois pare. Não tente traduzir o livro inteiro de uma
vez; o valor do processo está em blocos pequenos, consistentes e reproduzíveis.

## Regras invioláveis (nunca quebre)
1. **Escopo:** trabalhe **somente** dentro de `composite/traducao-pt-BR/` e
   **leia** `composite/composite_planets_restructured.md`. Não toque em mais
   nada do repositório. Este é um projeto isolado.
2. **Nunca edite o arquivo-fonte** em inglês.
3. **Nunca reescreva um bloco já `done`** nem retraduza o que outra sessão fez.
   Estilo divergente entre sessões é o principal risco — não o crie.
4. **Nunca edite `GLOSSARIO.md` no meio de uma tradução.** Termo novo vai para
   `GLOSSARIO-CANDIDATOS.md`. A promoção ao glossário é tarefa de curadoria.
5. **O glossário é lei.** Mesmo que você prefira outra palavra, use a travada.
   Consistência > preferência.
6. **Não resuma, não omita, não acrescente.** Cada parágrafo do fonte gera um
   parágrafo na tradução. Traduza o *sentido* em bom PT-BR, não ao pé da letra.
7. **Um bloco só é "done"** depois de: salvo em `blocos/NNN.md`, **verificado**
   (`ferramentas.py verificar NNN` aprovado) e **commitado + enviado (push)**.
   Se a sessão acabar antes disso, deixe o bloco como está — ele continua
   pendente para a próxima. Nunca marque `done` um bloco parcial.

## Fluxo obrigatório da sessão
1. `git pull` (ou fetch) na branch de trabalho — **comece sempre do estado mais
   recente**, senão você retraduz blocos já feitos ou gera conflito no MANIFEST.
2. Leia `GLOSSARIO.md` e `GUIA-DE-ESTILO.md` por inteiro (NÃO leia o fonte
   inteiro — ver "Dieta de contexto").
3. `python3 ferramentas.py continuar` → ele diz o bloco e a próxima ação
   (inclusive se há bloco órfão a recuperar).
4. (Se houver risco de sessões em paralelo) `python3 ferramentas.py reservar NNN
   <sua-sessao>` e **faça commit+push dessa reserva antes de traduzir**.
5. `python3 ferramentas.py extrair NNN` → traduza **exatamente** esse trecho.
6. Salve em `blocos/NNN.md` (só a tradução).
7. `python3 ferramentas.py verificar NNN` → corrija até **APROVADO**.
8. Registre termos novos em `GLOSSARIO-CANDIDATOS.md`.
9. `python3 ferramentas.py concluir NNN <sua-sessao>`.
10. Commit + push (`git push -u origin <branch>`). Mensagem:
    `traducao: bloco NNN (<titulo>) PT-BR`.

## Pontos de quebra conhecidos e como agir
- **Omissão/truncamento silencioso** (o pior risco num livro repetitivo): o
  `verificar` compara nº de títulos e de `<!-- image -->` (têm de bater exato) e
  a razão de palavras PT/EN (~1,1–1,3). Se **REPROVAR**, você pulou um título,
  uma interpretação ou um parágrafo — encontre e complete antes de concluir.
- **Bloco grande** (ex.: bloco 008, ~3.300 palavras): traduza-o **inteiro**. Se
  o orçamento da sessão apertar, é melhor **não concluir** do que entregar pela
  metade. Não fatie a saída de um bloco em arquivos separados.
- **Concorrência entre sessões:** duas sessões podem pegar o mesmo bloco. Sempre
  `git pull` no início; use `reservar` + push antes de traduzir; se o `push`
  for rejeitado, `git pull --rebase` e tente de novo. Se, após o pull, o bloco
  já estiver `done`/`in_progress` por outro, pegue o **próximo** pendente.
- **OCR ruim:** o fonte tem palavras coladas e trocadas. Corrija em silêncio
  quando o sentido é claro; só use `<!-- NOTA: -->` + campo `notes` do MANIFEST
  para trechos **genuinamente** ilegíveis. Nunca deixe lacuna.
- **Títulos de interpretação:** siga o padrão travado (GLOSSARIO §5/§6). Ex.:
  `### Conjunct Moon` → `### Conjunção com a Lua`. Não invente variações.
- **Referências cruzadas** ("see Chapter Four", "Figure 4", "page 26"): use os
  títulos de capítulo travados e **mantenha os números** do original.
- **Push falhou por rede:** tente de novo com espera crescente (2s, 4s, 8s, 16s)
  até 4 vezes. Só faça push para a branch de trabalho designada.

## Retomada entre sessões ("continue de onde parou")
O estado do projeto **não vive na sua memória** — vive no **git** (o `MANIFEST.csv`
diz o que está `done`/`in_progress`/`pending`, e `blocos/` guarda as traduções).
Qualquer sessão nova, em qualquer container, retoma assim:
1. `git pull` na branch de trabalho (traz o estado real e mais recente).
2. `python3 ferramentas.py continuar` — este comando **é** o bastão: ele diz
   quantos blocos faltam e **qual é a próxima ação exata**.
3. Faça o que ele mandar. Fim. Você não precisa de nenhum resumo da sessão
   anterior — o `continuar` reconstrói tudo sozinho.

**Sessão anterior que caiu no meio (bloco órfão `in_progress`):** o `continuar`
detecta e resolve:
- se já existe tradução válida do bloco → ele manda `verificar` + `concluir`;
- se a tradução está ausente/incompleta → ele manda você **assumir o bloco e
  traduzi-lo do zero** (ou `soltar NNN` para devolvê-lo à fila).
Nunca deixe um bloco preso em `in_progress` ao encerrar: ou conclua, ou `soltar`.

## Dieta de contexto (orçamento de tokens) — CRÍTICO
Cada sessão tem ~200k tokens. **Não desperdice nenhum** carregando o que não
precisa. A causa nº 1 de sessão "que já começa na metade do limite" é ler o
livro inteiro.
- **NUNCA** abra/`Read`/`cat` o arquivo-fonte `composite_planets_restructured.md`
  inteiro. Ele tem **~185 mil tokens** — sozinho quase enche a janela.
  **Sempre** obtenha o trecho do seu bloco com `python3 ferramentas.py extrair
  NNN` (~2,5k tokens).
- **NUNCA** leia `RESULTADO.md` (cresce sem limite) nem outros arquivos de
  `blocos/` "para pegar contexto". Você não precisa deles: a consistência vem do
  **glossário**, não de reler traduções passadas.
- **Kit de leitura da sessão (e só ele):** `CLAUDE.md` (~1,2k) + `GLOSSARIO.md`
  (~1,7k) + `GUIA-DE-ESTILO.md` (~1k) + o bloco via `extrair` (~2,5k). Total
  ~6–7k tokens. Sobra a janela inteira para traduzir com folga.
- Prefira os comandos do `ferramentas.py` a leituras amplas de arquivo: `proximo`,
  `continuar` e `verificar` devolvem só o essencial.
- Um bloco = uma sessão enxuta. Se quiser fazer vários, faça-os **em sequência**
  (traduz, conclui, dá push, e só então começa o próximo) — não carregue tudo de
  uma vez.

## Definição de pronto (Definition of Done) de um bloco
Traduzido integralmente • glossário respeitado • `verificar` APROVADO • termos
novos em CANDIDATOS • MANIFEST em `done` • commitado e enviado. Só então a
sessão terminou.
