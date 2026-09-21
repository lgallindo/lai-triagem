"""Importa o artefato treinado para o repositório de modelos do BentoML.

    cd ~/lai-triagem && uv run python scripts/register_bento.py

Toma artifacts/model_arrival.txt (texto nativo do LightGBM) mais
artifacts/preprocessor.json e registra os dois como um único modelo BentoML,
com o pré-processador embarcado em `custom_objects`, de modo que o serviço não
precisa de arquivos avulsos.
"""

import json
from pathlib import Path

import bentoml
import lightgbm as lgb

ART = Path(__file__).resolve().parents[1] / "artifacts"
NAME = "lai_triagem_arrival"

booster = lgb.Booster(model_file=str(ART / "model_arrival.txt"))
meta = json.loads((ART / "preprocessor.json").read_text(encoding="utf-8"))

saved = bentoml.lightgbm.save_model(
    NAME,
    booster,
    signatures={"predict": {"batchable": True, "batch_dim": 0}},
    # O pré-processador viaja com o modelo: evita divergência treino/serviço.
    custom_objects={"preprocessor": meta},
    labels={
        "task": "lai-reencaminhamento-risk",
        "snapshot": meta["data_snapshot"],
        "train_years": ",".join(map(str, meta["train_years"])),
    },
    metadata={
        "test_pr_auc_matured": meta["metrics"]["arrival"]["test_matured"]["pr_auc"],
        "test_precision_at_5pct":
            meta["metrics"]["arrival"]["test_matured"]["precision_at"]["0.05"],
        "n_features": len(meta["feature_order"]),
        "excluded_leakage_features": meta["excluded_leakage_features"],
        # Registrado no próprio artefato para que ninguém o implante sem saber.
        "caveat": ("modelo honesto de chegada; apenas empata com uma consulta "
                   "histórica por órgão. Ver docs/VERIFICATION.md"),
    },
)

print(f"registered: {saved.tag}")
print(f"  path     : {saved.path}")
print(f"  trees    : {booster.num_trees()}")
print(f"  features : {len(meta['feature_order'])}")
print("\nservir com:  uv run bentoml serve service.py:LaiTriagem")
