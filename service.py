"""BentoML service for LAI reencaminhamento-risk triage.

    cd ~/lai-triagem
    .venv/bin/python scripts/register_bento.py     # model.txt -> BentoML store
    .venv/bin/bentoml serve service.py:LaiTriagem

The artifact is a LightGBM native text model plus a JSON sidecar, so nothing
here depends on pickle compatibility.

Read docs/VERIFICATION.md before trusting the score: the honest arrival-time
model barely outperforms a one-line organ-rate lookup, which is also exposed
here as /score_baseline for comparison.
"""

from __future__ import annotations

from pathlib import Path

import bentoml
import lightgbm as lgb
from pydantic import BaseModel, Field

from lai_triagem.featurize import Preprocessor, risk_label

ART = Path(__file__).parent / "artifacts"
MODEL_TAG = "lai_triagem_arrival:latest"
THRESHOLD = 0.1691  # top-10% queue operating point


class PedidoLAI(BaseModel):
    """Arrival-time fields only. Post-hoc fields are rejected by the guard."""
    OrgaoDestinatario: str = Field(..., description="Organ the citizen addressed")
    DataRegistro: str = Field(..., description="DD/MM/YYYY")
    Esfera: str | None = None
    UF: str | None = None
    Municipio: str | None = None
    FormaResposta: str | None = None
    OrigemSolicitacao: str | None = None
    TipoDemandante: str | None = None
    Genero: str | None = None
    Escolaridade: str | None = None
    Profissao: str | None = None
    TipoPessoaJuridica: str | None = None
    Pais: str | None = None
    UF_sol: str | None = None
    Municipio_sol: str | None = None
    DataNascimento: str | None = None


@bentoml.service(name="lai_triagem", traffic={"timeout": 20},
                 resources={"cpu": "2"})
class LaiTriagem:
    bento_model = bentoml.models.get(MODEL_TAG)

    def __init__(self) -> None:
        self.booster: lgb.Booster = bentoml.lightgbm.load_model(self.bento_model)
        self.prep = Preprocessor(self.bento_model.custom_objects["preprocessor"])

    @bentoml.api
    def score(self, pedido: PedidoLAI) -> dict:
        payload = pedido.model_dump()
        X = self.prep.transform(payload)
        prob = float(self.booster.predict(X)[0])
        return {
            "probabilidade_reencaminhamento": round(prob, 6),
            "alerta": risk_label(prob, THRESHOLD),
            "threshold": THRESHOLD,
            "orgao_conhecido": self.prep.organ_is_known(payload["OrgaoDestinatario"]),
            "orgao_rate_historica": round(float(X["orgao_rate"].iloc[0]), 6),
            "base_rate_coorte": round(self.prep.base_rate, 6),
        }

    @bentoml.api
    def score_baseline(self, pedido: PedidoLAI) -> dict:
        """The organ-rate lookup, for comparison. On matured 2026 test data this
        scored precision@5% of 24.79% against the model's 24.44%."""
        payload = pedido.model_dump()
        rate = float(self.prep.organ_rate.get(payload["OrgaoDestinatario"],
                                              self.prep.base_rate))
        return {
            "probabilidade_reencaminhamento": round(rate, 6),
            "alerta": risk_label(rate, THRESHOLD),
            "metodo": "lookup histórico por órgão (sem modelo)",
            "orgao_conhecido": self.prep.organ_is_known(payload["OrgaoDestinatario"]),
        }

    @bentoml.api
    def health(self) -> dict:
        return {
            "model_tag": str(self.bento_model.tag),
            "n_trees": self.booster.num_trees(),
            "n_features": len(self.prep.feature_order),
            "features": self.prep.feature_order,
            "excluded_leakage_fields": sorted(self.prep.meta["excluded_leakage_features"]),
            "train_years": self.prep.meta["train_years"],
            "data_snapshot": self.prep.meta["data_snapshot"],
        }
