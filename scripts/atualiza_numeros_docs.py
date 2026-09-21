"""
Regenera no README os números que derivam do artefato.

Existe porque esses números já divergiram **quatro** vezes. Mantê-los à mão não
é sustentável: cada retreinamento move ~25 percentuais de ganho, três blocos de
resposta HTTP e os decimais citados em prosa. `check_prosa.py` acusa a
divergência com arquivo e linha; este script a corrige.

O que regenera, sempre lendo do artefato e nunca estimando:

  1. a coluna de ganho de toda tabela cujo cabeçalho fala de ganho;
  2. todo bloco ```json que segue um `curl`, pontuando o payload em processo
     pelo mesmo caminho que o serviço embrulha;
  3. os decimais de seis casas citados em prosa, trocando o valor antigo de
     cada cenário pelo novo.

O que **não** toca: prosa analítica, conclusões, e qualquer número que não
derive mecanicamente do artefato. Esses continuam sendo decisão de quem escreve
— e `check_prosa.py` continua sendo o juiz.

    uv run python scripts/atualiza_numeros_docs.py            # aplica
    uv run python scripts/atualiza_numeros_docs.py --ensaio   # só mostra
"""

import json
import re
import sys
from pathlib import Path

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lai_triagem.featurize import Preprocessor, risk_label  # noqa: E402

ENSAIO = "--ensaio" in sys.argv
ART = ROOT / "artifacts"
README = ROOT / "README.md"

prep = Preprocessor.from_json(ART / "preprocessor.json")
booster = lgb.Booster(model_file=str(ART / "model_arrival.txt"))
g = booster.feature_importance(importance_type="gain")
GANHO = {n: 100.0 * v / float(g.sum()) for n, v in zip(booster.feature_name(), g)}

texto = README.read_text(encoding="utf-8")
mudancas = []


def pontua(payload, baseline=False):
    limpo = {k: v for k, v in payload.items() if v is not None}
    if baseline:
        taxa = float(prep.organ_rate.get(limpo["OrgaoDestinatario"], prep.base_rate))
        return {"probabilidade_reencaminhamento": round(taxa, 6),
                "alerta": risk_label(taxa, prep.threshold),
                "metodo": "lookup histórico por órgão (sem modelo)",
                "orgao_conhecido": prep.organ_is_known(limpo["OrgaoDestinatario"])}
    X = prep.transform(limpo)
    p = float(booster.predict(X)[0])
    cal = prep.calibrate(p)
    return {"probabilidade_reencaminhamento": round(p, 6),
            "alerta": risk_label(p, prep.threshold),
            "threshold": prep.threshold,
            "probabilidade_calibrada": None if cal is None else round(cal, 6),
            "calibrada_apenas_para_leitura": True,
            "orgao_conhecido": prep.organ_is_known(limpo["OrgaoDestinatario"]),
            "orgao_rate_historica": round(float(X["orgao_rate"].iloc[0]), 6),
            "orgao_rate_movel_90d": round(float(X["orgao_rate_movel_90d"].iloc[0]), 6),
            "base_rate_coorte": round(prep.base_rate, 6),
            "historico_informado": prep.history_supplied(limpo),
            "historico_parcial_ignorado": prep.history_partial_ignored(limpo)}


# ------------------------------------------------- 1. colunas de ganho
linhas = texto.splitlines(keepends=True)
em_tabela = False
for i, linha in enumerate(linhas):
    if not linha.startswith("|"):
        em_tabela = False
        continue
    if "Ganho" in linha or "Uso no modelo" in linha:
        em_tabela = True
        continue
    if not em_tabela or "baixo" in linha or "ausente" in linha:
        continue
    m = re.match(r"^\|\s*`([^`]+)`", linha)
    if not m or m.group(1) not in GANHO:
        continue
    pct = re.search(r"(\d+,\d+)%", linha)
    if not pct:
        continue
    novo = f"{GANHO[m.group(1)]:.2f}".replace(".", ",")
    if pct.group(1) != novo:
        linhas[i] = linha[:pct.start(1)] + novo + linha[pct.end(1):]
        mudancas.append(f"ganho de `{m.group(1)}`: {pct.group(1)}% -> {novo}%")
texto = "".join(linhas)

# --------------------------------------- 2. blocos de resposta e 3. prosa
RX_CURL = re.compile(r"curl[^\n]*?-d '(\{.*?\})'", re.S)
RX_JSON = re.compile(r"```json\n(.*?)\n```", re.S)
trocas_prosa = {}

for m in list(RX_CURL.finditer(texto)):
    corpo = json.loads(m.group(1))
    if "pedido" not in corpo:
        continue
    real = pontua(corpo["pedido"], baseline="/score_baseline" in m.group(0))
    seguinte = RX_JSON.search(texto, m.end())
    if not seguinte or "probabilidade_reencaminhamento" not in seguinte.group(1):
        continue
    antigo = json.loads(seguinte.group(1))
    if antigo == real:
        continue
    va, vn = antigo["probabilidade_reencaminhamento"], real["probabilidade_reencaminhamento"]
    if va != vn:
        trocas_prosa[f"{va:.6f}".replace(".", ",")] = f"{vn:.6f}".replace(".", ",")
    novo_bloco = "```json\n" + json.dumps(real, indent=2, ensure_ascii=False) + "\n```"
    texto = texto[:seguinte.start()] + novo_bloco + texto[seguinte.end():]
    mudancas.append(f"bloco de resposta: {va} -> {vn}")

for antigo, novo in trocas_prosa.items():
    if antigo in texto:
        texto = texto.replace(antigo, novo)
        mudancas.append(f"prosa: {antigo} -> {novo}")

# ---------------------------------------------------------------------------
if not mudancas:
    print("nada a atualizar — o README já está coerente com o artefato")
    sys.exit(0)

print(f"{len(mudancas)} atualizacao(oes){' (ENSAIO, nada gravado)' if ENSAIO else ''}:")
for c in mudancas:
    print(f"  · {c}")
if not ENSAIO:
    README.write_text(texto, encoding="utf-8")
    print("\nREADME.md gravado. Confira com: uv run python scripts/check_prosa.py")
