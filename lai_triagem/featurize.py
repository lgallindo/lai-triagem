"""Arrival-time featurisation, shared by training and serving.

Single source of truth so the service cannot drift from the trained model.
Depends only on pandas/numpy plus the JSON sidecar -- no pickles, so there is
no scikit-learn/pandas version coupling at load time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Fields that exist only AFTER triage or after the response. Rejected on input.
# Each entry was established empirically -- see docs/VERIFICATION.md.
LEAKAGE_FIELDS = {
    "Situacao", "FoiProrrogado", "FoiReencaminhado", "DataResposta", "Decisao",
    "EspecificacaoDecisao", "DetalhamentoDecisao", "MotivoNegativaAcesso",
    "PrazoRestricaoAcesso", "AssuntoPedido", "SubAssuntoPedido", "Tag",
    "PrazoAtendimento",  # rewritten on prorrogation: +10d per LAI art.11 par.2
}


class Preprocessor:
    """Turns one arrival-time request dict into the model's feature row."""

    def __init__(self, meta: dict):
        self.meta = meta
        self.feature_order: list[str] = meta["feature_order"]
        self.categorical: list[str] = meta["categorical_features"]
        self.numeric: list[str] = meta["numeric_features"]
        self.codes: dict[str, dict[str, int]] = meta["category_codes"]
        self.organ_rate: dict[str, float] = meta["organ_rate"]
        self.base_rate: float = meta["base_rate"]

    @classmethod
    def from_json(cls, path: str | Path) -> "Preprocessor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def check_no_leakage(self, payload: dict) -> None:
        bad = LEAKAGE_FIELDS.intersection(payload)
        if bad:
            raise ValueError(
                f"post-hoc field(s) supplied, refusing to score: {sorted(bad)}. "
                "These are unavailable when a request arrives; see docs/VERIFICATION.md."
            )

    def transform(self, payload: dict) -> pd.DataFrame:
        self.check_no_leakage(payload)
        p = {k: (v.strip() if isinstance(v, str) else v) for k, v in payload.items()}

        reg = pd.to_datetime(p.get("DataRegistro"), format="%d/%m/%Y", errors="coerce")
        nasc = pd.to_datetime(p.get("DataNascimento"), format="%d/%m/%Y", errors="coerce")

        idade = np.nan
        if pd.notna(reg) and pd.notna(nasc):
            idade = round((reg - nasc).days / 365.25, 1)
            if not (10 <= idade <= 110):
                idade = np.nan

        organ = p.get("OrgaoDestinatario")
        derived = {
            "reg_month": reg.month if pd.notna(reg) else np.nan,
            "reg_dow": reg.dayofweek if pd.notna(reg) else np.nan,
            "reg_day": reg.day if pd.notna(reg) else np.nan,
            "idade": idade,
            "orgao_rate": float(self.organ_rate.get(organ, self.base_rate)),
            "uf_match": int((p.get("UF_sol") or "~") == (p.get("UF") or "!")),
        }

        row: dict[str, object] = {}
        for c in self.categorical:
            # Unseen or absent level -> -1, which LightGBM treats as missing.
            row[c] = self.codes.get(c, {}).get(p.get(c), -1)
        for c in self.numeric:
            row[c] = derived.get(c, p.get(c, np.nan))

        return pd.DataFrame([row], columns=self.feature_order).astype("float64")

    def organ_is_known(self, organ: str | None) -> bool:
        return organ in self.organ_rate


def risk_label(prob: float, threshold: float) -> str:
    return "ALTO RISCO" if prob >= threshold else "BAIXO RISCO"
