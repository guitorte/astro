# GUIA DE ESTILO — Tradução PT-BR de *Planets in Composite*

Leia isto **inteiro** antes de traduzir qualquer bloco. Junto com o
`GLOSSARIO.md`, é o que mantém a tradução idêntica em espírito de uma sessão
para outra.

## 1. Registro e voz
- **Português do Brasil**, natural e fluente — traduza o *sentido*, não
  palavra por palavra. Evite decalque do inglês (ordem de frase, voz passiva
  excessiva, "isto significa que…").
- Tratamento: **"você" / "vocês"** (nunca "tu"). O livro se dirige a **um
  casal**, então o "you" quase sempre é **plural → "vocês"**. Só use "você"
  (singular) quando o texto fala de um indivíduo genérico.
- Tom: reflexivo, aconselhador e respeitoso, como o original de Hand. Nem
  acadêmico demais, nem coloquial demais. Mantenha a seriedade técnica da
  astrologia sem soar como manual de autoajuda.
- Gênero: o casal é genérico. Prefira construções neutras ("o casal", "a
  dupla", "vocês", "a pessoa parceira") quando natural; não force flexões
  masculinas/femininas onde o original é neutro.

## 2. Fidelidade
- **Traduza tudo**: não resuma, não omita, não acrescente. Cada parágrafo do
  original vira um parágrafo correspondente.
- Preserve a estrutura de parágrafos e a ordem das ideias.
- Preserve ênfases (itálico/negrito) quando existirem no original.

## 3. Markdown e elementos não textuais — PRESERVAR NA POSIÇÃO
- Mantenha os níveis de título (`#`, `##`, `###`) exatamente como no fonte —
  só traduza o **texto** do título (ver padrões no GLOSSARIO seções 5 e 6).
- Mantenha `<!-- image -->` **exatamente onde está**, sem traduzir.
- Tabelas: preserve a sintaxe `|...|`. Traduza o conteúdo das células; mantenha
  números, páginas e a formatação da tabela.
- Não altere números de figura, página, capítulo ou dados de mapas.

## 4. Erros de OCR (o fonte foi escaneado)
O texto-fonte tem MUITO ruído de OCR: palavras coladas ("youwill",
"wilbe", "tô", "byy"), letras trocadas ("reporduced", "Merucry", "Camposite",
"signficant", "thunb-indexed"), acentos estranhos ("cłosely"), espaços
faltando.
- **Corrija silenciosamente** quando o sentido é óbvio — a tradução deve sair
  em português correto e limpo. Ex.: "youwill accomplish" → "vocês vão
  realizar".
- Se um trecho estiver **genuinamente ambíguo/ilegível**, faça sua melhor
  interpretação e registre a dúvida no campo `notes` do MANIFEST (e/ou um
  comentário `<!-- NOTA: ... -->` no bloco). Não deixe lacuna em branco.
- A página de rosto, ficha catalográfica e os formulários rasurados do fim do
  livro (linhas finais: "AM PM", "Staie", "Statr"…) podem ser traduzidos de
  forma aproximada/limpa; sinalize em `notes` se algo for irrecuperável.

## 5. Termos técnicos
- Siga o `GLOSSARIO.md` **à risca**. Em caso de dúvida, o glossário vence.
- Termo recorrente que falta no glossário → escolha uma opção, use-a de forma
  consistente e **anote em `GLOSSARIO-CANDIDATOS.md`** (bloco, EN, PT sugerido).
  Assim a próxima sessão herda a mesma decisão.

## 6. Formato de saída de cada bloco
- Salve em `blocos/NNN.md` (ex.: `blocos/007.md`), só a tradução — sem
  cabeçalhos extras, sem o texto em inglês, sem comentários de processo (exceto
  `<!-- NOTA: -->` quando necessário).
- O conteúdo do bloco deve começar e terminar nos mesmos títulos/parágrafos do
  trecho-fonte correspondente, para que a concatenação final fique perfeita.

## 7. Checagem final antes de marcar "done" (2 minutos)
1. Todos os títulos do trecho-fonte estão presentes e traduzidos pelo padrão?
2. Nenhum parágrafo foi pulado (conte os parágrafos do fonte x tradução)?
3. Todos os termos do glossário foram respeitados (Sol, Lua, casas, aspectos)?
4. `<!-- image -->` e tabelas preservados na posição?
5. Português flui e está sem resíduo de OCR?
6. Termos novos registrados em `GLOSSARIO-CANDIDATOS.md`?
