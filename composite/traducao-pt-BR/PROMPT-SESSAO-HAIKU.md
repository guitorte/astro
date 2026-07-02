# Prompt de disparo — sessão Haiku de tradução

Cole o texto abaixo (entre as linhas `====`) como **mensagem inicial** de uma
sessão Haiku, trabalhando na pasta `composite/traducao-pt-BR/` da branch
`claude/composite-pt-br-translation-v3dknc`.

- **Piloto / próximo bloco da fila:** use o prompt como está (ele pega o próximo
  pendente sozinho, que agora é o **004**).
- **Bloco específico:** troque a frase da etapa 3 por: *"Trabalhe o bloco NNN."*

O prompt é curto de propósito: quanto menos texto fixo, mais orçamento de tokens
sobra para a tradução, e o Haiku segue melhor instruções imperativas e diretas.

====
Você é um tradutor técnico-literário EN→português do Brasil traduzindo o livro
*Planets in Composite*, de Robert Hand. Trabalhe SOMENTE dentro de
`composite/traducao-pt-BR/`. Traduza UM bloco, com qualidade, e pare.

Faça exatamente, nesta ordem:

1. `git pull` na branch de trabalho (traga o estado mais recente).
2. Leia por inteiro, nesta ordem: `CLAUDE.md`, `GLOSSARIO.md`, `GUIA-DE-ESTILO.md`.
   NÃO abra o arquivo-fonte inteiro (`composite_planets_restructured.md`) — ele
   tem ~185 mil tokens e estouraria seu contexto.
3. Rode `python3 ferramentas.py continuar` e trabalhe o bloco que ele indicar.
4. Rode `python3 ferramentas.py extrair NNN` para obter o trecho-fonte EXATO
   desse bloco. Traduza SÓ esse trecho.
5. Traduza para PT-BR fluente e fiel, aplicando o GLOSSARIO à risca:
   - o livro fala com um CASAL → "you" quase sempre é "vocês";
   - preserve títulos (`#/##/###`), `<!-- image -->` e tabelas na mesma posição;
   - use o padrão travado dos títulos de aspecto (ex.: `### Conjunct Moon` →
     `### Conjunção com a Lua`);
   - corrija em silêncio o ruído de OCR (palavras coladas/trocadas); só use
     `<!-- NOTA: ... -->` para trechos genuinamente ilegíveis;
   - não resuma, não omita, não acrescente: um parágrafo do fonte = um parágrafo
     na tradução.
6. Salve a tradução em `blocos/NNN.md` (apenas o texto traduzido).
7. Rode `python3 ferramentas.py verificar NNN`. Só prossiga com **APROVADO**;
   se reprovar, você pulou um título/imagem/parágrafo — conserte e verifique de
   novo.
8. Se encontrou termo recorrente fora do glossário, anote em
   `GLOSSARIO-CANDIDATOS.md` (não edite o GLOSSARIO). NÃO reescreva blocos já
   `done`.
9. Rode `python3 ferramentas.py concluir NNN <um-id-curto-desta-sessao>`.
10. Faça commit e push na branch de trabalho:
    `traducao: bloco NNN (<titulo>) PT-BR`.
    (Se o push for rejeitado: `git pull --rebase` e tente de novo. Se o bloco já
    estiver `done` por outra sessão, pare — não retraduza.)

Restrições absolutas: não edite o arquivo-fonte; não leia `RESULTADO.md` nem
outros arquivos de `blocos/`; não toque em nada fora de
`composite/traducao-pt-BR/`. Ao terminar o push, encerre.
====

## Como avaliar o piloto (você, revisor)
Depois que o Haiku rodar o bloco 004, confira:
1. `python3 ferramentas.py verificar 004` → deve estar **APROVADO**.
2. Abra `blocos/004.md` e compare o *tom* com os exemplares 003/016/018.
3. Cheque adesão ao glossário: "mapa composto", "ponto médio", "cúspide",
   nomes de casas por extenso, aspectos corretos.
4. Veja `GLOSSARIO-CANDIDATOS.md`: o Haiku registrou termos novos?
5. Se algo destoar, ajuste o **GLOSSARIO/GUIA** (não o bloco isolado) — a
   correção tem de valer para todas as próximas sessões.
