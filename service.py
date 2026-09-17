"""Serviço BentoML de triagem de risco de reencaminhamento de pedidos LAI.

    cd ~/lai-triagem
    uv run python scripts/register_bento.py     # model.txt -> repositório BentoML
    uv run bentoml serve service.py:LaiTriagem

O artefato é um modelo em texto nativo do LightGBM mais um acompanhante JSON,
portanto nada aqui depende de compatibilidade de `pickle`.

CONTRATO DE DADOS. As variáveis de histórico do solicitante são **opcionais e
fornecidas pelo chamador**: são dado pessoal e o artefato não as retém. O
Fala.BR já as possui. Omitidas, o escore opera degradado e `/score` sinaliza
isso em `historico_informado`. Racional completo em
docs/DECISAO_ESTADO_SOLICITANTE.md.

Leia docs/VERIFICATION.md antes de confiar no escore. /score_baseline expõe a
consulta histórica por órgão para comparação permanente.
"""

from __future__ import annotations

from pathlib import Path

import bentoml
import lightgbm as lgb
from pydantic import BaseModel, Field

from lai_triagem.featurize import Preprocessor, risk_label

ART = Path(__file__).parent / "artifacts"
MODEL_TAG = "lai_triagem_arrival:latest"


class PedidoLAI(BaseModel):
    """Somente campos de chegada. Campos posteriores são rejeitados pela barreira."""

    # --- obrigatórios ------------------------------------------------------
    OrgaoDestinatario: str = Field(..., description="Órgão a que o cidadão endereçou")
    DataRegistro: str = Field(..., description="DD/MM/AAAA")

    # --- opcionais, do pedido e do perfil ----------------------------------
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

    # --- histórico do solicitante: FORNECIDO PELO CHAMADOR -----------------
    # Dado pessoal. O artefato não o retém; o Fala.BR já o possui. Omitir
    # qualquer um deles é permitido e apenas reduz a precisão.
    n_pedidos_previos: float | None = Field(
        None, description="Quantos pedidos este solicitante já fez antes deste")
    prev_reenc_solicitante: float | None = Field(
        None, description="Quantos dos pedidos anteriores foram reencaminhados")
    prev_reenc_rate_solicitante: float | None = Field(
        None, description="Razão entre os dois anteriores")
    n_pedidos_previos_neste_orgao: float | None = Field(
        None, description="Pedidos anteriores deste solicitante A ESTE órgão")
    prev_reenc_neste_orgao: float | None = Field(
        None, description="Reencaminhamentos anteriores deste solicitante neste órgão")
    n_orgaos_distintos_previos: float | None = Field(
        None, description="Quantos órgãos distintos este solicitante já acionou")
    dias_desde_ultimo_pedido: float | None = Field(
        None, description="Dias desde o pedido anterior deste solicitante")


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
        """Escore do modelo, com a taxa do órgão exposta para auditoria."""
        payload = {k: v for k, v in pedido.model_dump().items() if v is not None}
        X = self.prep.transform(payload)
        prob = float(self.booster.predict(X)[0])
        cal = self.prep.calibrate(prob)
        return {
            # Escore que ORDENA a fila e define o alerta. É o cru.
            "probabilidade_reencaminhamento": round(prob, 6),
            "alerta": risk_label(prob, self.prep.threshold),
            "threshold": self.prep.threshold,
            # Só para leitura humana: interpretável como probabilidade, mas NÃO
            # ordena. A isotônica introduz empates e custou 0,56 pp de
            # precisão@5% quando usada para ranquear.
            "probabilidade_calibrada": None if cal is None else round(cal, 6),
            "calibrada_apenas_para_leitura": True,
            "orgao_conhecido": self.prep.organ_is_known(payload["OrgaoDestinatario"]),
            "orgao_rate_historica": round(float(X["orgao_rate"].iloc[0]), 6),
            "orgao_rate_movel_90d": round(float(X["orgao_rate_movel_90d"].iloc[0]), 6),
            "base_rate_coorte": round(self.prep.base_rate, 6),
            # Transparência sobre operação degradada: sem histórico do
            # solicitante o modelo perde precisão de forma mensurável.
            "historico_informado": self.prep.history_supplied(payload),
        }

    @bentoml.api
    def score_baseline(self, pedido: PedidoLAI) -> dict:
        """A consulta por órgão, sem modelo, mantida para comparação permanente.
        No teste maturado de 2026 obteve precisão@5% de 24,79% contra 30,99%
        do modelo."""
        payload = pedido.model_dump()
        rate = float(self.prep.organ_rate.get(payload["OrgaoDestinatario"],
                                              self.prep.base_rate))
        return {
            "probabilidade_reencaminhamento": round(rate, 6),
            "alerta": risk_label(rate, self.prep.threshold),
            "metodo": "lookup histórico por órgão (sem modelo)",
            "orgao_conhecido": self.prep.organ_is_known(payload["OrgaoDestinatario"]),
        }

    @bentoml.api
    def health(self) -> dict:
        """Procedência do artefato, incluindo o contrato de dados e as exclusões."""
        m = self.prep.meta
        return {
            "model_tag": str(self.bento_model.tag),
            "n_trees": self.booster.num_trees(),
            "n_features": len(self.prep.feature_order),
            "features": self.prep.feature_order,
            "threshold": self.prep.threshold,
            "queue_fraction": m.get("queue_fraction"),
            # O integrador descobre o contrato sem ler o código.
            "caller_supplied_features": self.prep.caller_features,
            "caller_supplied_default": self.prep.caller_default,
            "caller_supplied_rationale": m.get("caller_supplied_rationale"),
            "embedded_organ_tables": [k for k in ("organ_rate", "organ_rate_movel_90d",
                                                  "organ_rate_movel_365d", "organ_birth")
                                      if k in m],
            "contains_personal_data": False,
            "excluded_leakage_fields": sorted(m["excluded_leakage_features"]),
            "train_years": m["train_years"],
            "data_snapshot": m["data_snapshot"],
        }
