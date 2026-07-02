#!/usr/bin/env python3
"""Ferramentas de apoio a traducao por blocos.

Uso:
  python3 ferramentas.py proximo            -> mostra o proximo bloco 'pending'
  python3 ferramentas.py extrair 007        -> imprime o texto-fonte EXATO do bloco 007
  python3 ferramentas.py extrair 007 > x    -> salva o trecho-fonte para conferencia
  python3 ferramentas.py status             -> painel de progresso
  python3 ferramentas.py montar             -> concatena blocos prontos em RESULTADO.md
  python3 ferramentas.py concluir 007 sessao-haiku-A  -> marca bloco como done
  python3 ferramentas.py verificar 007      -> checa paridade estrutural fonte x traducao
  python3 ferramentas.py reservar 007 sessao-haiku-A  -> marca 'in_progress' (claim)
  python3 ferramentas.py continuar          -> "continue de onde parou": estado + acao
  python3 ferramentas.py soltar 007         -> devolve bloco travado a fila (pending)
"""
import csv, os, sys, re

BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, "..", "composite_planets_restructured.md")
MANI = os.path.join(BASE, "MANIFEST.csv")

def load():
    with open(MANI, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save(rows):
    fields = ["block_id","status","src_start_line","src_end_line","word_count",
              "primary_title","session_by","output_file","notes"]
    with open(MANI, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def src_lines():
    return open(SRC, encoding="utf-8").read().split("\n")

def extrair(bid):
    rows = load()
    r = next((x for x in rows if x["block_id"] == bid), None)
    if not r: sys.exit(f"bloco {bid} nao encontrado")
    L = src_lines()
    a, b = int(r["src_start_line"]), int(r["src_end_line"])
    return "\n".join(L[a-1:b])

def proximo():
    rows = load()
    for r in rows:
        if r["status"] == "pending":
            print(f"PROXIMO BLOCO: {r['block_id']}  ({r['word_count']} palavras)  "
                  f"linhas {r['src_start_line']}-{r['src_end_line']}")
            print(f"titulo: {r['primary_title']}")
            print(f"saida:  {r['output_file']}")
            return
    print("Nenhum bloco pendente. Traducao completa!")

def status():
    rows = load()
    done = [r for r in rows if r["status"] == "done"]
    pend = [r for r in rows if r["status"] == "pending"]
    wdone = sum(int(r["word_count"]) for r in done)
    wtot  = sum(int(r["word_count"]) for r in rows)
    print(f"Blocos: {len(done)}/{len(rows)} concluidos "
          f"({len(pend)} pendentes)")
    print(f"Palavras: {wdone}/{wtot} ({100*wdone//max(wtot,1)}%)")

def _parity_ok(bid, r):
    """True/False se a traducao existe e bate estruturalmente; None se nao existe."""
    p = os.path.join(BASE, r["output_file"])
    if not os.path.exists(p): return None
    sh, si, _, sw = _metrics(extrair(bid))
    th, ti, _, tw = _metrics(open(p, encoding="utf-8").read())
    ratio = tw / sw if sw else 0
    return (sh == th and si == ti and 0.9 <= ratio <= 1.6)

def continuar():
    """Ponto unico de retomada: 'continue de onde parou'.
    Reconstroi o estado a partir do MANIFEST (que veio do git) e diz a acao."""
    rows = load()
    done = [r for r in rows if r["status"] == "done"]
    prog = [r for r in rows if r["status"] == "in_progress"]
    pend = [r for r in rows if r["status"] == "pending"]
    wdone = sum(int(r["word_count"]) for r in done)
    wtot  = sum(int(r["word_count"]) for r in rows)
    print("== RETOMADA ==")
    print(f"Concluidos {len(done)}/{len(rows)} blocos "
          f"({100*wdone//max(wtot,1)}% das palavras). Pendentes: {len(pend)}.")

    # 1) Blocos orfaos (reservados por sessao anterior que pode ter caido)
    for r in prog:
        bid = r["block_id"]
        ok = _parity_ok(bid, r)
        if ok is True:
            print(f"\n[ORFAO PRONTO] bloco {bid} estava 'in_progress' e ja tem "
                  f"traducao valida — so falta fechar:")
            print(f"  python3 ferramentas.py verificar {bid}")
            print(f"  python3 ferramentas.py concluir {bid} <sua-sessao>")
            return
        else:
            motivo = "sem arquivo de traducao" if ok is None else "traducao incompleta (paridade falhou)"
            print(f"\n[ORFAO INTERROMPIDO] bloco {bid} reservado por "
                  f"'{r['session_by']}' porem {motivo}. A sessao anterior caiu.")
            print(f"  -> Assuma este bloco: traduza-o do zero (linhas "
                  f"{r['src_start_line']}-{r['src_end_line']}).")
            print(f"     python3 ferramentas.py extrair {bid}")
            print(f"     (ou 'soltar {bid}' para devolve-lo a fila de pendentes)")
            return

    # 2) Sem orfaos -> proximo pendente
    if pend:
        r = pend[0]
        print(f"\nPROXIMO BLOCO: {r['block_id']}  ({r['word_count']} palavras)  "
              f"linhas {r['src_start_line']}-{r['src_end_line']}  | {r['primary_title']}")
        print("  python3 ferramentas.py extrair " + r["block_id"])
        return
    print("\nTraducao COMPLETA. Rode: python3 ferramentas.py montar")

def soltar(bid):
    """Devolve um bloco 'in_progress' a fila de pendentes (sessao anterior caiu)."""
    rows = load()
    for r in rows:
        if r["block_id"] == bid:
            if r["status"] == "done":
                sys.exit(f"bloco {bid} esta 'done' — nao solte.")
            r["status"] = "pending"; r["session_by"] = ""
    save(rows)
    print(f"bloco {bid} devolvido a fila (pending).")

def montar():
    rows = load()
    out = []
    for r in rows:
        if r["status"] != "done": break
        p = os.path.join(BASE, r["output_file"])
        if not os.path.exists(p):
            sys.exit(f"faltando arquivo pronto: {p}")
        out.append(open(p, encoding="utf-8").read().rstrip())
    dst = os.path.join(BASE, "RESULTADO.md")
    open(dst, "w", encoding="utf-8").write("\n\n".join(out) + "\n")
    print(f"Montado {len(out)} blocos em {dst}")

def reservar(bid, quem):
    rows = load()
    for r in rows:
        if r["block_id"] == bid:
            if r["status"] == "done":
                sys.exit(f"bloco {bid} ja esta 'done' — nao reserve.")
            r["status"] = "in_progress"; r["session_by"] = quem
    save(rows)
    print(f"bloco {bid} reservado (in_progress) por {quem}. "
          f"Faca commit+push desta reserva ANTES de traduzir se houver sessoes paralelas.")

def _metrics(text):
    heads = len(re.findall(r'(?m)^#{1,3}\s', text))
    imgs  = text.count("<!-- image -->")
    paras = len([p for p in re.split(r'\n\s*\n', text) if p.strip()])
    words = len(re.findall(r'\S+', text))
    return heads, imgs, paras, words

def verificar(bid):
    rows = load()
    r = next((x for x in rows if x["block_id"] == bid), None)
    if not r: sys.exit(f"bloco {bid} nao encontrado")
    p = os.path.join(BASE, r["output_file"])
    if not os.path.exists(p):
        sys.exit(f"traducao ausente: {p}")
    sh, si, sp, sw = _metrics(extrair(bid))
    th, ti, tp, tw = _metrics(open(p, encoding="utf-8").read())
    ratio = tw / sw if sw else 0
    ok = True
    print(f"bloco {bid} — fonte x traducao")
    def line(name, a, b, hard=False):
        nonlocal ok
        flag = "OK " if a == b else ("ERRO" if hard else "aviso")
        if a != b and hard: ok = False
        print(f"  [{flag}] {name}: {a} -> {b}")
    line("titulos (#/##/###)", sh, th, hard=True)
    line("marcadores <!-- image -->", si, ti, hard=True)
    line("paragrafos (aprox.)", sp, tp)
    print(f"  [{'OK ' if 0.9 <= ratio <= 1.6 else 'aviso'}] "
          f"palavras: {sw} -> {tw} (razao PT/EN {ratio:.2f}; esperado ~1.1-1.3)")
    if not (0.9 <= ratio <= 1.6):
        print("      razao fora da faixa: possivel OMISSAO (baixa) ou EXCESSO (alta).")
    print("RESULTADO:", "APROVADO (paridade estrutural ok)" if ok
          else "REPROVADO — titulos/imagens divergem: reveja antes de concluir.")

def concluir(bid, quem):
    rows = load()
    for r in rows:
        if r["block_id"] == bid:
            r["status"] = "done"; r["session_by"] = quem
    save(rows)
    print(f"bloco {bid} marcado como done por {quem}")

if __name__ == "__main__":
    if len(sys.argv) < 2: print(__doc__); sys.exit()
    cmd = sys.argv[1]
    if cmd == "extrair": print(extrair(sys.argv[2]))
    elif cmd == "proximo": proximo()
    elif cmd == "status": status()
    elif cmd == "montar": montar()
    elif cmd == "concluir": concluir(sys.argv[2], sys.argv[3] if len(sys.argv)>3 else "")
    elif cmd == "reservar": reservar(sys.argv[2], sys.argv[3] if len(sys.argv)>3 else "")
    elif cmd == "verificar": verificar(sys.argv[2])
    elif cmd == "continuar": continuar()
    elif cmd == "soltar": soltar(sys.argv[2])
    else: print(__doc__)
