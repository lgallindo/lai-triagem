"""EXEMPLO MÍNIMO — pontua um pedido LAI. Não requer BentoML.

    cd ~/lai-triagem && uv run python examples/minimal_predict.py

Carrega o modelo em texto nativo do LightGBM mais o acompanhante JSON, pontua um
pedido com dados de chegada e imprime o sinalizador ALTO/BAIXO RISCO. É
exatamente o caminho de código que o serviço BentoML envolve: se isto funciona,
o serviço também funciona.
"""

import sys
from pathlib import Path

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lai_triagem.featurize import Preprocessor, risk_label  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"

booster = lgb.Booster(model_file=str(ART / "model_arrival.txt"))
prep = Preprocessor.from_json(ART / "preprocessor.json")

# Tudo aqui é conhecido no momento em que o pedido entra na caixa do Fala.BR.
request = {
    "OrgaoDestinatario": "CC-PR – Casa Civil da Presidência da República",
    "Esfera": "Federal",
    "UF": " ",
    "Municipio": " ",
    "FormaResposta": "Pelo sistema (com avisos por email)",
    "OrigemSolicitacao": "Internet",
    "DataRegistro": "15/09/2026",
    "TipoDemandante": "Pessoa Física",
    "Genero": "Masculino",
    "Escolaridade": "Ensino Fundamental",
    "Profissao": "Outra",
    "TipoPessoaJuridica": None,
    "Pais": "Brasil",
    "UF_sol": "PE",
    "Municipio_sol": "Recife",
    "DataNascimento": "08/10/1976",
}

# Ponto de operação da fila de 10%; ver artifacts/preprocessor.json.
THRESHOLD = 0.1691

X = prep.transform(request)
prob = float(booster.predict(X)[0])

print(f"órgão                 : {request['OrgaoDestinatario']}")
print(f"órgão conhecido       : {prep.organ_is_known(request['OrgaoDestinatario'])}")
print(f"taxa histórica        : {X['orgao_rate'].iloc[0]:.4f}   (taxa-base {prep.base_rate:.4f})")
print(f"P(reencaminhamento)   : {prob:.4f}")
print(f"sinalizador           : {risk_label(prob, THRESHOLD)}   (limiar {THRESHOLD})")

# Um órgão de competência estreita deve pontuar muito mais baixo.
# Verificação de sanidade, não teste automatizado.
low = dict(request, OrgaoDestinatario="UFLA – Universidade Federal de Lavras")
p_low = float(booster.predict(prep.transform(low))[0])
print(f"\ncontraste, UFLA       : {p_low:.4f}  ->  {risk_label(p_low, THRESHOLD)}")

# Fornecer campo posterior à triagem deve ser recusado, não pontuado em silêncio.
try:
    prep.transform(dict(request, FoiProrrogado="Sim"))
except ValueError as e:
    print(f"\nbarreira de vazamento : OK — {e}")
