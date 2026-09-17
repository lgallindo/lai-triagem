"""Serviço BentoML de triagem de risco de reencaminhamento de pedidos LAI.

    cd ~/lai-triagem
    uv run python scripts/register_bento.py     # model.txt -> repositório BentoML
    uv run bentoml serve service.py:LaiTriagem

O artefato é um modelo em texto nativo do LightGBM mais um acompanhante JSON,
portanto nada aqui depende de compatibilidade de `pickle`.

Leia docs/VERIFICATION.md antes de confiar no escore: o modelo honesto de
chegada apenas empata com uma consulta histórica por órgão de uma única linha,
que também é exposta aqui em /score_baseline para comparação.
"""

from __future__ import annotations

from pathlib import Path

import bentoml
import lightgbm as lgb
from pydantic import BaseModel, Field

from lai_triagem.featurize import Preprocessor, risk_label

ART = Path(__file__).parent / "artifacts"
MODEL_TAG = "lai_triagem_arrival:latest"

# Ponto de operação da fila de 10%, não probabilidade calibrada.
THRESHOLD = 0.1691


class PedidoLAI(BaseModel):
    """Somente campos de chegada. Campos posteriores são rejeitados pela barreira."""
    OrgaoDestinatario: str = Field(..., description="Órgão a que o cidadão endereçou")
    DataRegistro: str = Field(..., description="DD/MM/AAAA")
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
    bento_model = bentoml.models.BentoModel(MODEL_TAG)

    def __init__(self) -> None:
        self.booster: lgb.Booster = bentoml.lightgbm.load_model(self.bento_model)
        # O pré-processador vem embarcado no próprio modelo registrado.
        self.prep = Preprocessor(self.bento_model.custom_objects["preprocessor"])

    @bentoml.api
    def score(self, pedido: PedidoLAI) -> dict:
        """Escore do modelo treinado, com a taxa do órgão exposta para auditoria."""
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
        """A consulta por órgão, para comparação. No teste maturado de 2026 obteve
        precisão@5% de 24,79% contra 24,44% do modelo."""
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
        """Procedência do artefato, incluindo o que foi deliberadamente excluído."""
        return {
            "model_tag": str(self.bento_model.tag),
            "n_trees": self.booster.num_trees(),
            "n_features": len(self.prep.feature_order),
            "features": self.prep.feature_order,
            "excluded_leakage_fields": sorted(self.prep.meta["excluded_leakage_features"]),
            "train_years": self.prep.meta["train_years"],
            "data_snapshot": self.prep.meta["data_snapshot"],
        }
