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
DOCS = [p for p in list(ROOT.glob("*.md")) + list(ROOT.glob("docs/*.md"))
        if p.name != "METRICAS.md"]

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

# METRICAS.md tem de existir e ser mais novo que o artefato.
mdoc = ROOT / "docs/METRICAS.md"
if not mdoc.exists():
    falhas.append("docs/METRICAS.md ausente — rode scripts/train.py")
elif mdoc.stat().st_mtime < (ROOT / "artifacts/preprocessor.json").stat().st_mtime - 5:
    falhas.append("docs/METRICAS.md mais antigo que o artefato — regere")

if falhas:
    print(f"\nFALHOU: {len(falhas)} divergência(s)")
    for f in falhas:
        print(f"  {f}")
    sys.exit(1)
print("\nOK — nenhum documento contradiz o artefato")
