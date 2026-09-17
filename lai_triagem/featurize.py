"""Featurização no instante da chegada, compartilhada por treino e serviço.

Fonte única da verdade, para que o serviço não possa divergir do modelo
treinado. Depende apenas de pandas/numpy mais o acompanhante JSON — sem
`pickle`, logo não há acoplamento de versão no carregamento.

As variáveis separam-se por natureza do titular do dado, conforme
docs/DECISAO_ESTADO_SOLICITANTE.md:

  * lado do ÓRGÃO   — embarcado no artefato (conduta de entidade pública):
                      orgao_rate, orgao_rate_movel_90d/_365d, idade do órgão
  * lado do SOLICITANTE — FORNECIDO PELO CHAMADOR, nunca retido. São dado
                      pessoal; o Fala.BR já os possui. Ausentes, valem -1, que
                      o LightGBM trata como faltante, exatamente como no
                      treinamento para solicitante anonimizado.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Campos que só existem DEPOIS da triagem ou da resposta. Rejeitados na entrada.
# Cada item foi estabelecido empiricamente — ver docs/CAMPOS_POST_HOC.md.
LEAKAGE_FIELDS = {
    "Situacao", "FoiProrrogado", "FoiReencaminhado", "DataResposta", "Decisao",
    "EspecificacaoDecisao", "DetalhamentoDecisao", "MotivoNegativaAcesso",
    "PrazoRestricaoAcesso", "AssuntoPedido", "SubAssuntoPedido", "Tag",
    # reescrito na prorrogação: +10 dias pelo art. 11 §2 da LAI
    "PrazoAtendimento", "prazo_dias",
    # codifica a unidade registradora, não um sequencial neutro (H5)
    "ProtocoloPedido", "protocolo_seq",
}


class Preprocessor:
    """Converte um pedido (dicionário) na linha de variáveis do modelo."""

    def __init__(self, meta: dict):
        self.meta = meta
        self.feature_order: list[str] = meta["feature_order"]
        self.categorical: list[str] = meta["categorical_features"]
        self.numeric: list[str] = meta["numeric_features"]
        self.codes: dict[str, dict[str, int]] = meta["category_codes"]
        self.organ_rate: dict[str, float] = meta["organ_rate"]
        self.organ_movel_90: dict[str, float] = meta.get("organ_rate_movel_90d", {})
        self.organ_movel_365: dict[str, float] = meta.get("organ_rate_movel_365d", {})
        self.organ_birth: dict[str, str] = meta.get("organ_birth", {})
        self.base_rate: float = meta["base_rate"]
        self.threshold: float = meta.get("threshold", 0.1691)
        # Nomes que o chamador deve informar; ausentes viram o valor padrão.
        self.caller_features: list[str] = meta.get("caller_supplied_features", [])
        self.caller_default: float = float(meta.get("caller_supplied_default", -1))

    @classmethod
    def from_json(cls, path: str | Path) -> "Preprocessor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def check_no_leakage(self, payload: dict) -> None:
        """Recusa a requisição se ela trouxer qualquer campo posterior à triagem."""
        bad = LEAKAGE_FIELDS.intersection(payload)
        if bad:
            raise ValueError(
                f"post-hoc field(s) supplied, refusing to score: {sorted(bad)}. "
                "These are unavailable when a request arrives; see docs/CAMPOS_POST_HOC.md."
            )

    def transform(self, payload: dict) -> pd.DataFrame:
        self.check_no_leakage(payload)
        p = {k: (v.strip() if isinstance(v, str) else v) for k, v in payload.items()}

        reg = pd.to_datetime(p.get("DataRegistro"), format="%d/%m/%Y", errors="coerce")
        nasc = pd.to_datetime(p.get("DataNascimento"), format="%d/%m/%Y", errors="coerce")

        idade = np.nan
        if pd.notna(reg) and pd.notna(nasc):
            idade = round((reg - nasc).days / 365.25, 1)
            # Idades implausíveis são erro de dado, não sinal.
            if not (10 <= idade <= 110):
                idade = np.nan

        organ = p.get("OrgaoDestinatario")
        # Idade do órgão: dias entre o registro e a primeira aparição do órgão.
        idade_orgao = np.nan
        born = self.organ_birth.get(organ)
        if born and pd.notna(reg):
            idade_orgao = float((reg - pd.Timestamp(born)).days)

        derived = {
            "reg_month": reg.month if pd.notna(reg) else np.nan,
            "reg_dow": reg.dayofweek if pd.notna(reg) else np.nan,
            "reg_day": reg.day if pd.notna(reg) else np.nan,
            "idade": idade,
            # Órgão não visto no treino recai na taxa-base da coorte.
            "orgao_rate": float(self.organ_rate.get(organ, self.base_rate)),
            "orgao_rate_movel_90d": float(self.organ_movel_90.get(organ, self.base_rate)),
            "orgao_rate_movel_365d": float(self.organ_movel_365.get(organ, self.base_rate)),
            "dias_desde_primeiro_pedido_do_orgao": idade_orgao,
            "uf_match": int((p.get("UF_sol") or "~") == (p.get("UF") or "!")),
        }
        # Histórico do solicitante: só do chamador. Nunca lido de tabela interna.
        for c in self.caller_features:
            v = p.get(c)
            derived[c] = self.caller_default if v is None else float(v)

        row: dict[str, object] = {}
        for c in self.categorical:
            # Nível inédito ou ausente vira -1, que o LightGBM trata como faltante.
            row[c] = self.codes.get(c, {}).get(p.get(c), -1)
        for c in self.numeric:
            row[c] = derived.get(c, p.get(c, np.nan))

        return pd.DataFrame([row], columns=self.feature_order).astype("float64")

    def organ_is_known(self, organ: str | None) -> bool:
        """Se falso, o escore recai na taxa-base e não deve ser lido como sinal."""
        return organ in self.organ_rate

    def history_supplied(self, payload: dict) -> bool:
        """O chamador informou algum contador de histórico? Se não, o modelo
        opera degradado — mensurável, mas silencioso se não for exposto."""
        return any(payload.get(c) is not None for c in self.caller_features)


def risk_label(prob: float, threshold: float) -> str:
    """Rótulo operacional. O limiar é ponto de operação de fila, não probabilidade."""
    return "ALTO RISCO" if prob >= threshold else "BAIXO RISCO"
