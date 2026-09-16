"""MINIMAL EXAMPLE — score one LAI request. No BentoML needed.

    cd ~/lai-triagem && .venv/bin/python examples/minimal_predict.py

Loads the LightGBM native text model plus the JSON sidecar, scores one
arrival-time request, and prints the ALTO/BAIXO RISCO flag. This is the exact
code path the BentoML service wraps, so if this works the service will too.
"""

import sys
from pathlib import Path

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lai_triagem.featurize import Preprocessor, risk_label  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"

booster = lgb.Booster(model_file=str(ART / "model_arrival.txt"))
prep = Preprocessor.from_json(ART / "preprocessor.json")

# Everything here is known the moment the request lands in the Fala.BR inbox.
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

THRESHOLD = 0.1691  # top-10% operating point; see artifacts/preprocessor.json

X = prep.transform(request)
prob = float(booster.predict(X)[0])

print(f"organ            : {request['OrgaoDestinatario']}")
print(f"organ known      : {prep.organ_is_known(request['OrgaoDestinatario'])}")
print(f"historical rate  : {X['orgao_rate'].iloc[0]:.4f}   (cohort base {prep.base_rate:.4f})")
print(f"P(reencaminhado) : {prob:.4f}")
print(f"flag             : {risk_label(prob, THRESHOLD)}   (threshold {THRESHOLD})")

# A narrowly-scoped organ should score far lower -- sanity check, not a test.
low = dict(request, OrgaoDestinatario="UFLA – Universidade Federal de Lavras")
p_low = float(booster.predict(prep.transform(low))[0])
print(f"\ncontrast, UFLA   : {p_low:.4f}  ->  {risk_label(p_low, THRESHOLD)}")

# Supplying a post-hoc field must be refused rather than silently scored.
try:
    prep.transform(dict(request, FoiProrrogado="Sim"))
except ValueError as e:
    print(f"\nleakage guard    : OK — {e}")
