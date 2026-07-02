#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monta um dicionario JSON de consulta a partir dos blocos ja traduzidos.

Percorre os blocos em ordem de fonte (contiguos), mantendo o contexto de
planeta/secao entre blocos, e agrupa cada verbete (casa ou aspecto) sob seu
planeta. Conteudo de dicionario = Capitulo Quatro (As Casas + Ascendente) e os
capitulos de planetas (Sol, Lua, Mercurio, Venus...).
"""
import re
import json
from collections import OrderedDict
from pathlib import Path

BASE = Path("/home/user/astro/composite/traducao-pt-BR")
BLOCOS = BASE / "blocos"
# Blocos de conteudo de dicionario, em ordem de fonte:
IDS = [f"{n:03d}" for n in range(13, 37)]

data = OrderedDict()

def planeta(p):
    if p not in data:
        data[p] = {
            "significado": "",
            "casas": OrderedDict(),
            "aspectos": OrderedDict(),
        }
    return data[p]

def normaliza_planeta(p):
    return p.strip()

def classifica_h1(h):
    t = h.strip()
    m = re.search(r":\s*(.+)$", t)
    nome = (m.group(1) if m else t).strip()
    if nome == "As Casas":
        return "As Casas"
    return re.sub(r"^(O |A |Os |As )", "", nome).strip()

def classifica_h2(h):
    """Retorna (tipo, planeta, extra)."""
    t = h.strip()
    if t == "Significado das Casas":
        return ("sig", "As Casas", None)
    if t == "O Ascendente Composto":
        return ("ascendente", "As Casas", None)
    m = re.match(r"^(?:O |A )?Significado (?:do|da|de|dos|das) (.+?)(?: no Mapa Composto)?$", t)
    if m:
        return ("sig", normaliza_planeta(m.group(1)), None)
    m = re.match(r"^(?:O |A )?(.+?) nas Casas$", t)
    if m:
        return ("casas", normaliza_planeta(m.group(1)), None)
    m = re.match(r"^Aspectos (?:do|da|de) (.+)$", t)
    if m:
        return ("aspectos", normaliza_planeta(m.group(1)), None)
    # anomalia OCR: "Sol Composto em Conjuncao com Netuno Composto"
    m = re.match(r"^(.+?) Composto em (Conjunção|Sextil|Quadratura|Trígono|Oposição) (?:com|a|à|ao) (.+?)(?: Composto)?$", t)
    if m:
        chave = f"{m.group(2)} com {m.group(3).strip()}"
        return ("aspect_entry", normaliza_planeta(m.group(1)), chave)
    return (None, None, None)

# Estado global
cur_planet = None
as_casas_mode = False
cur_section = None          # 'sig' | 'casas' | 'aspectos'
target = None               # tupla descrevendo destino do texto acumulado
buf = []

def limpa(linhas):
    txt = "\n".join(linhas)
    txt = re.sub(r"<!--\s*image\s*-->", "", txt)
    paras = [p.strip() for p in re.split(r"\n\s*\n", txt)]
    paras = [p for p in paras if p]
    return "\n\n".join(paras)

def flush():
    global buf, target
    if target is None:
        buf = []
        return
    texto = limpa(buf)
    buf = []
    if not texto:
        return
    kind = target[0]
    if kind == "sig":
        planeta(target[1])["significado"] = texto
    elif kind == "casas_entry":
        planeta(target[1])["casas"][target[2]] = texto
    elif kind == "aspectos_entry":
        planeta(target[1])["aspectos"][target[2]] = texto
    elif kind == "posicao":
        planeta("As Casas")["casas"][target[2]] = texto
    elif kind == "casas_intro":
        planeta(target[1])["casas_intro"] = texto
    elif kind == "aspectos_intro":
        planeta(target[1])["aspectos_intro"] = texto

for bid in IDS:
    for raw in (BLOCOS / f"{bid}.md").read_text(encoding="utf-8").splitlines():
        m1 = re.match(r"^#\s+(.*)$", raw)
        m2 = re.match(r"^##\s+(.*)$", raw)
        m3 = re.match(r"^###\s+(.*)$", raw)
        if m1:
            flush()
            cur_planet = classifica_h1(m1.group(1))
            as_casas_mode = (cur_planet == "As Casas")
            planeta(cur_planet)
            cur_section = None
            target = None
            continue
        if m2:
            kind, pl, extra = classifica_h2(m2.group(1))
            # Cabecalho de secao duplicado (artefato de OCR): ignora, para que o
            # verbete pendente continue acumulando seu corpo.
            if kind in ("sig", "casas", "aspectos") and pl == cur_planet and cur_section == kind:
                continue
            flush()
            if kind == "sig":
                cur_planet = pl
                as_casas_mode = (pl == "As Casas")
                cur_section = "sig"
                target = ("sig", pl)
            elif kind == "casas":
                cur_planet = pl
                as_casas_mode = False
                cur_section = "casas"
                target = ("casas_intro", pl)
            elif kind == "aspectos":
                cur_planet = pl
                as_casas_mode = False
                cur_section = "aspectos"
                target = ("aspectos_intro", pl)
            elif kind == "ascendente":
                cur_planet = "As Casas"
                as_casas_mode = True
                cur_section = "casas"
                target = ("posicao", "As Casas", "O Ascendente Composto")
            elif kind == "aspect_entry":
                target = ("aspectos_entry", pl, extra)
            elif as_casas_mode:
                # verbete de casa (Cap. Quatro usa ## para as casas)
                target = ("posicao", "As Casas", m2.group(1).strip())
            else:
                target = None
            continue
        if m3:
            flush()
            titulo = m3.group(1).strip()
            if cur_section == "casas":
                target = ("casas_entry", cur_planet, titulo)
            elif cur_section == "aspectos":
                target = ("aspectos_entry", cur_planet, titulo)
            else:
                target = None
            continue
        buf.append(raw)

flush()

# --- Monta saida limpa e ordenada ---
ordem = ["As Casas", "Sol", "Lua", "Mercúrio", "Vênus"]
saida = OrderedDict()

meta = OrderedDict()
meta["fonte"] = "Planets in Composite — Robert Hand"
meta["traducao"] = "português do Brasil (projeto composite/traducao-pt-BR)"
meta["descricao"] = ("Dicionário de consulta com os textos completos já traduzidos: "
                     "significado de posições (casas), dos planetas e dos aspectos.")
meta["gerado_de_blocos"] = "013–036 (linhas 800–2593 da fonte)"
meta["observacoes"] = [
    "Marcadores de imagem (<!-- image -->) foram removidos; o texto é fiel à tradução.",
    "Em 'As Casas' estão o significado geral de cada casa e o Ascendente Composto.",
    "Em cada planeta: 'significado' (texto introdutório), 'casas' (planeta em cada casa) e 'aspectos' (aspectos com outros pontos).",
    "A anomalia de OCR '## Sol Composto em Conjunção com Netuno Composto' foi normalizada para o aspecto 'Conjunção com Netuno' do Sol.",
    "Vênus está incompleto na fonte traduzida até aqui (apenas o significado do planeta).",
]

cobertura = OrderedDict()
saida["_meta"] = meta

for p in ordem + [k for k in data if k not in ordem]:
    if p not in data:
        continue
    d = data[p]
    ent = OrderedDict()
    if d.get("significado"):
        ent["significado"] = d["significado"]
    if d.get("casas_intro"):
        ent["casas_intro"] = d["casas_intro"]
    if d["casas"]:
        rot = "posicoes" if p == "As Casas" else "casas"
        ent[rot] = d["casas"]
    if d.get("aspectos_intro"):
        ent["aspectos_intro"] = d["aspectos_intro"]
    if d["aspectos"]:
        ent["aspectos"] = d["aspectos"]
    saida[p] = ent
    cobertura[p] = OrderedDict([
        ("tem_significado", bool(d.get("significado"))),
        ("casas", len(d["casas"])),
        ("aspectos", len(d["aspectos"])),
    ])

meta["cobertura"] = cobertura

out = BASE / "DICIONARIO-CONSULTA.json"
out.write_text(json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# Resumo no stdout
print("Gerado:", out)
for p, c in cobertura.items():
    print(f"  {p:10s} significado={c['tem_significado']!s:5s} casas={c['casas']:2d} aspectos={c['aspectos']:2d}")
