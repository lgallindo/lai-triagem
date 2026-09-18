"""
Fix 6 — impede que a documentação volte a contradizer o artefato.

Sai com código 1 se algum `.md` citar tamanho de modelo, limiar ou contagem de
campos divergente de `artifacts/preprocessor.json`. A causa dos números
obsoletos era eu repetir métricas em prosa em quatro documentos; `METRICAS.md`
passou a ser a fonte gerada, e este script é o guarda.

    uv run python scripts/check_docs_numbers.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / "artifacts/preprocessor.json").read_text(encoding="utf-8"))

N_VARS = len(meta["feature_order"])
THRESHOLD = meta["threshold"]
N_CALLER = len(meta["caller_supplied_features"])
N_EXCL = len(meta["excluded_leakage_features"])

# `METRICAS.md` é gerado e por definição coerente; não se audita a si mesmo.
# A varredura de `docs/` é RECURSIVA de propósito: quando as auditorias foram
# movidas para `docs/auditorias/`, um glob não recursivo as teria tirado do
# alcance do guarda sem ninguém notar. Omissão silenciosa é pior que isenção
# declarada -- se algum arquivo precisar de isenção, que ela esteja escrita.
# `docs/auditorias/` também é isento, e por motivo diferente: registro de
# auditoria CITA o estado do artefato no dia em que foi feito. "limiar 0,163204"
# num relatório de 18/09/2026 está correto como história e passaria a falhar no
# próximo retreinamento. Exatamente como `METRICAS.md`, a isenção é declarada e
# impressa, não silenciosa.
DOCS = [p for p in list(ROOT.glob("*.md")) + list(ROOT.glob("docs/**/*.md"))
        if p.name != "METRICAS.md" and "auditorias" not in p.parts]

# Padrões que carregam um número verificável, com o valor esperado.
REGRAS = [
    (re.compile(r"\*\*(\d+)\s+vari[áa]veis\*\*"), N_VARS, "nº de variáveis"),
    (re.compile(r"(\d+)\s+vari[áa]veis,\s*\d+\s+[áa]rvores"), N_VARS, "nº de variáveis"),
    (re.compile(r"limiar\s+\*?\*?(\d+[.,]\d+)"), THRESHOLD, "limiar"),
    (re.compile(r"\*\*(\d+)\*\*\s+campos\s+enviados"), N_CALLER, "campos do chamador"),
]

falhas = []
for doc in sorted(DOCS):
    texto = doc.read_text(encoding="utf-8")
    for linha_n, linha in enumerate(texto.splitlines(), 1):
        # Trechos que declaram explicitamente serem históricos ficam isentos.
        if any(m in linha for m in ("anteriores ao Fix", "já foi muito melhor",
                                    "versões anteriores", "Versões anteriores",
                                    "antes do Fix", "era vazamento")):
            continue
        for rx, esperado, rotulo in REGRAS:
            for achado in rx.findall(linha):
                val = float(achado.replace(",", ".")) if "." in achado or "," in achado \
                    else int(achado)
                if isinstance(esperado, float):
                    ok = abs(val - esperado) < 1e-6
                else:
                    ok = int(val) == esperado
                if not ok:
                    falhas.append(f"{doc.relative_to(ROOT)}:{linha_n}  {rotulo}: "
                                  f"documento diz {achado}, artefato diz {esperado}")

print(f"artefato: {N_VARS} variáveis, limiar {THRESHOLD}, "
      f"{N_CALLER} campos do chamador, {N_EXCL} exclusões")
print(f"documentos auditados: {len(DOCS)}")

# METRICAS.md tem de existir e descrever ESTE artefato.
#
# Antes isto comparava mtime, e mtime não é conteúdo: `git checkout`, um clone
# ou uma cópia reordenam mtimes à vontade. Pior, o protocolo de auditoria deste
# projeto restaura `artifacts/` depois de cada rodada -- ou seja, a própria
# rotina de auditoria fazia este guarda falhar por engano, com o conteúdo
# perfeitamente coerente. A comparação agora é do carimbo que o METRICAS.md
# declara contra o `created_utc` gravado no artefato: exata, e imune a mtime.
mdoc = ROOT / "docs/METRICAS.md"
if not mdoc.exists():
    falhas.append("docs/METRICAS.md ausente — rode scripts/train.py")
else:
    carimbo_artefato = meta.get("created_utc")
    # Ancorado no `Z` final: sem isso a classe de caracteres engole o ponto
    # que encerra a frase, e o carimbo nunca casa.
    achado = re.search(r"Gerado por `scripts/train\.py` em ([0-9T:.\-]+Z)",
                       mdoc.read_text(encoding="utf-8"))
    if not achado:
        falhas.append("docs/METRICAS.md não declara quando foi gerado")
    elif carimbo_artefato and achado.group(1) != carimbo_artefato:
        falhas.append(
            f"docs/METRICAS.md descreve o treino de {achado.group(1)}, mas o "
            f"artefato é de {carimbo_artefato} — regere com scripts/train.py")

if falhas:
    print(f"\nFALHOU: {len(falhas)} divergência(s)")
    for f in falhas:
        print(f"  {f}")
    sys.exit(1)
print("\nOK — nenhum documento contradiz o artefato")
