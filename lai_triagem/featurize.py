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
        # O limiar vem do artefato (quantil da validação, recalibrado a cada
        # retreinamento). O padrão só existe para artefatos antigos sem a chave.
        # Sem fallback numérico: limiar fixo no código envelhece em
        # silêncio a cada retreinamento. Artefato sem a chave é inválido.
        self.threshold: float = float(meta["threshold"])
        # Nomes que o chamador deve informar; ausentes viram o valor padrão.
        self.caller_features: list[str] = meta.get("caller_supplied_features", [])
        self.caller_default: float = float(meta.get("caller_supplied_default", -1))
        # Derivadas aritmeticamente das de cima; o chamador não as envia.
        self.derived_from_caller: list[str] = meta.get("derived_from_caller", [])
        self.calibration: dict = meta.get("calibration") or {}

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
        # Histórico do solicitante: só do chamador, nunca de tabela interna.
        #
        # Fix 7 -- CONJUNTO ATÔMICO. Antes, enviar um único campo devolvia
        # `historico_informado: true` com os outros seis em -1: combinação que
        # nunca ocorre no treinamento (80 pedidos anteriores implicam razão
        # real), logo uma linha fora da distribuição. Agora, histórico parcial é
        # tratado como AUSENTE por inteiro, e o serviço sinaliza que ignorou.
        presentes = [c for c in self.caller_features if p.get(c) is not None]
        self._hist_parcial_ignorado = 0 < len(presentes) < len(self.caller_features)
        completo = len(presentes) == len(self.caller_features)
        for c in self.caller_features:
            derived[c] = float(p[c]) if completo else self.caller_default

        # As duas razões são DERIVADAS aqui, nunca aceitas do chamador, para não
        # poderem ficar incoerentes com o numerador e o denominador. Usam o
        # denominador MADURO, como no treinamento após o Fix 1+2.
        for nome, num, den in (
            ("prev_reenc_rate_solicitante", "prev_reenc_solicitante",
             "prev_reenc_solicitante_den"),
            ("prev_reenc_rate_neste_orgao", "prev_reenc_neste_orgao",
             "prev_reenc_neste_orgao_den"),
        ):
            if nome not in self.derived_from_caller:
                continue
            r, d = derived.get(num, self.caller_default), derived.get(den, self.caller_default)
            derived[nome] = (r / d if (d is not None and d > 0 and r is not None and r >= 0)
                             else self.caller_default)

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
        """O chamador informou o histórico COMPLETO? Fix 7: parcial conta como
        ausente, porque uma linha meio preenchida fica fora da distribuição de
        treinamento. Use `history_partial_ignored` para distinguir os casos."""
        return all(payload.get(c) is not None for c in self.caller_features)

    def history_partial_ignored(self, payload: dict) -> bool:
        """Verdadeiro quando o chamador mandou histórico incompleto e ele foi
        descartado por inteiro. Sinaliza erro de integração, não ausência."""
        n = sum(payload.get(c) is not None for c in self.caller_features)
        return 0 < n < len(self.caller_features)


    def calibrate(self, prob: float) -> float | None:
        """Escore calibrado (isotônica ajustada na validação), APENAS PARA LEITURA.

        NÃO use para ranquear nem para comparar com o limiar. A isotônica tem
        regiões planas, e dentro delas a ordenação original é destruída: medido,
        isso custou 0,56 pp de precisão@5% no teste maturado. O escore cru é que
        ordena a fila; o calibrado serve para o analista ler um número que
        significa probabilidade.
        """
        if not self.calibration:
            return None
        return float(np.interp(prob, self.calibration["grid"], self.calibration["image"]))


def risk_label(prob: float, threshold: float) -> str:
    """Rótulo operacional. O limiar é ponto de operação de fila, não probabilidade."""
    return "ALTO RISCO" if prob >= threshold else "BAIXO RISCO"
