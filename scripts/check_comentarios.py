"""
Camada de auditoria #15 — docstrings e comentários.

A regra do projeto é: documentação, docstrings e comentários em pt_BR; nomes de
arquivo, classe, função e variável podem ficar em en_US. Esta camada confere que
a regra vale, e que os comentários não apontam para coisas que deixaram de
existir.

O que se confere:

  1. Idioma: nenhuma docstring nem comentário predominantemente em inglês. A
     detecção é por contagem de palavras funcionais, sem dependência externa --
     um comentário com "the/of/and/is" e sem "de/que/não/para" está em inglês.
  2. Caminhos citados existem. Comentário que manda ler um arquivo movido ou
     apagado é pior que comentário nenhum, porque custa a busca.
  3. Referências obsoletas conhecidas: campo que deixou de ser entrada do
     chamador, defeito já corrigido ainda descrito como pendente.
  4. Relata, sem reprovar, as cadeias de `print()` em en_US nos scripts de
     auditoria e varredura -- decisão consciente de quem manda no projeto,
     registrada para não passar por esquecimento.

    uv run python scripts/check_comentarios.py
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FUNCIONAIS_PT = {
    "de", "que", "não", "nao", "para", "com", "uma", "por", "como", "mas", "ao",
    "dos", "das", "na", "no", "se", "é", "são", "foi", "tem", "já", "também",
    "ou", "sem", "sobre", "entre", "cada", "quando", "onde", "porque", "então",
    "isso", "este", "esta", "os", "as", "um", "do", "da", "em", "e", "a", "o",
    "pelo", "pela", "ser", "está", "seria", "porém", "aqui", "todo", "toda",
}
FUNCIONAIS_EN = {
    "the", "of", "and", "to", "in", "is", "that", "for", "with", "not", "are",
    "this", "be", "as", "by", "from", "it", "on", "was", "which", "has", "have",
    "but", "or", "an", "they", "their", "would", "should", "when", "where",
    "these", "those", "there", "been", "were", "will", "can", "could",
}

# Palavras de código que aparecem em texto pt_BR sem que ele seja inglês.
RX_PALAVRA = re.compile(r"[A-Za-zÀ-ÿ]+")

falhas: list[str] = []
avisos: list[str] = []
notas: list[str] = []


def idioma(texto: str):
    """Devolve (n_pt, n_en) contando palavras funcionais de cada idioma."""
    palavras = [p.lower() for p in RX_PALAVRA.findall(texto)]
    return (sum(p in FUNCIONAIS_PT for p in palavras),
            sum(p in FUNCIONAIS_EN for p in palavras))


def parece_ingles(texto: str) -> bool:
    if len(texto.strip()) < 40:        # curto demais para decidir
        return False
    pt, en = idioma(texto)
    return en >= 3 and en > pt


ARQUIVOS = sorted(
    [p for p in ROOT.rglob("*.py")
     if ".venv" not in p.parts and "__pycache__" not in p.parts])

RX_CAMINHO = re.compile(r"\b((?:docs|scripts|examples|lai_triagem|artifacts)/[\w./-]+\.(?:md|py|json|txt))")

OBSOLETOS = [
    (re.compile(r"prev_reenc_rate_\w+.{0,40}(informad|enviad|chamador)"),
     "descreve uma razão derivada como se fosse enviada pelo chamador"),
    (re.compile(r"AUDITORIA_EXTERNA\.md"),
     "aponta para docs/AUDITORIA_EXTERNA.md, que virou docs/auditorias/"),
    (re.compile(r"corre[çc][ãa]o pendente"),
     "diz 'correção pendente'; confirme que não foi corrigida"),
]

n_docstrings = n_comentarios = 0

for arq in ARQUIVOS:
    rel = arq.relative_to(ROOT)
    fonte = arq.read_text(encoding="utf-8")

    # ---- docstrings, via AST
    try:
        arvore = ast.parse(fonte)
    except SyntaxError as e:
        falhas.append(f"{rel}: não compila, {e}")
        continue
    for no in ast.walk(arvore):
        if not isinstance(no, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
            continue
        ds = ast.get_docstring(no)
        if not ds:
            continue
        n_docstrings += 1
        onde = getattr(no, "name", "<módulo>")
        linha = getattr(no, "lineno", 1)
        if parece_ingles(ds):
            pt, en = idioma(ds)
            falhas.append(f"{rel}:{linha}  docstring de `{onde}` parece inglês "
                          f"(pt={pt}, en={en}): {ds.strip()[:70]!r}")
        textos = [(ds, f"docstring de `{onde}`", linha)]
        for texto, rotulo, ln in textos:
            for achado in RX_CAMINHO.findall(texto):
                if not (ROOT / achado).exists():
                    falhas.append(f"{rel}:{ln}  {rotulo} cita `{achado}`, "
                                  f"que não existe")
            for rx, porque in OBSOLETOS:
                if rx.search(texto):
                    alvo = avisos if "confirme" in porque else falhas
                    alvo.append(f"{rel}:{ln}  {rotulo}: {porque}")

    # ---- comentários, via tokenize
    try:
        for tok in tokenize.generate_tokens(io.StringIO(fonte).readline):
            if tok.type != tokenize.COMMENT:
                continue
            txt = tok.string.lstrip("#").strip()
            if not txt or txt.startswith(("!", "-*-", "noqa", "type:")):
                continue
            n_comentarios += 1
            ln = tok.start[0]
            if parece_ingles(txt):
                pt, en = idioma(txt)
                falhas.append(f"{rel}:{ln}  comentário parece inglês "
                              f"(pt={pt}, en={en}): {txt[:70]!r}")
            for achado in RX_CAMINHO.findall(txt):
                if not (ROOT / achado).exists():
                    falhas.append(f"{rel}:{ln}  comentário cita `{achado}`, "
                                  f"que não existe")
            for rx, porque in OBSOLETOS:
                if rx.search(txt):
                    alvo = avisos if "confirme" in porque else falhas
                    alvo.append(f"{rel}:{ln}  comentário: {porque}")
    except (tokenize.TokenError, IndentationError) as e:
        avisos.append(f"{rel}: tokenize não terminou ({e})")

# ---- 4. cadeias de print() em en_US: relatado, não reprovado
en_prints: dict[str, int] = {}
for arq in ARQUIVOS:
    rel = str(arq.relative_to(ROOT))
    try:
        arvore = ast.parse(arq.read_text(encoding="utf-8"))
    except SyntaxError:
        continue
    for no in ast.walk(arvore):
        if (isinstance(no, ast.Call) and isinstance(no.func, ast.Name)
                and no.func.id == "print"):
            for a in no.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    if parece_ingles(a.value):
                        en_prints[rel] = en_prints.get(rel, 0) + 1
if en_prints:
    notas.append("cadeias de print() em en_US, decisão registrada de manter: "
                 + ", ".join(f"{k} ({v})" for k, v in sorted(en_prints.items())))

# ---------------------------------------------------------------------------
print(f"arquivos .py varridos: {len(ARQUIVOS)}")
print(f"docstrings conferidas: {n_docstrings}   comentários: {n_comentarios}")
for n in notas:
    print(f"  · {n}")
if avisos:
    print(f"\n{len(avisos)} aviso(s), não bloqueiam:")
    for a in avisos:
        print(f"  ~ {a}")
if falhas:
    print(f"\nFALHOU: {len(falhas)} problema(s) em docstring ou comentário")
    for f in falhas:
        print(f"  ✗ {f}")
    sys.exit(1)
print("\nOK — docstrings e comentários em pt_BR, sem caminho morto")
