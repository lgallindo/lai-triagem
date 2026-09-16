"""Import the trained artifact into the BentoML model store.

    cd ~/lai-triagem && .venv/bin/python scripts/register_bento.py

Takes artifacts/model_arrival.txt (LightGBM native text) plus
artifacts/preprocessor.json and registers them as one BentoML model, with the
preprocessor carried in custom_objects so the service needs no side files.
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
    custom_objects={"preprocessor": meta},
    labels={
        "task": "lai-reencaminhamento-risk",
        "snapshot": meta["data_snapshot"],
        "train_years": ",".join(map(str, meta["train_years"])),
    },
    metadata={
        "test_pr_auc_matured": meta["metrics"]["arrival"]["test_matured"]["pr_auc"],
        "test_precision_at_5pct": meta["metrics"]["arrival"]["test_matured"]["precision_at"]["0.05"],
        "n_features": len(meta["feature_order"]),
        "excluded_leakage_features": meta["excluded_leakage_features"],
        "caveat": ("honest arrival-time model; barely beats an organ-rate lookup. "
                   "See docs/VERIFICATION.md"),
    },
)

print(f"registered: {saved.tag}")
print(f"  path     : {saved.path}")
print(f"  trees    : {booster.num_trees()}")
print(f"  features : {len(meta['feature_order'])}")
print("\nserve with:  .venv/bin/bentoml serve service.py:LaiTriagem")
