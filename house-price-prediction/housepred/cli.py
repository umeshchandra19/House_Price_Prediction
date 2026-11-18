"""Command line:  python -m housepred train | predict"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import DEFAULT_DATA, ROOT

MODEL_PATH = ROOT / "models" / "best_model.joblib"
EXAMPLE = dict(longitude=-118.25, latitude=34.05, housing_median_age=30.0, total_rooms=2500.0,
               total_bedrooms=500.0, population=1300.0, households=480.0, median_income=4.2,
               ocean_proximity="<1H OCEAN")


def cmd_train(a):
    from .train import run
    run(a.data, a.results_dir, a.models_dir, seed=a.seed, drop_capped=a.drop_capped, quick=a.quick)


def cmd_predict(a):
    if not Path(a.model).exists():
        raise SystemExit("No trained model found. Run:  python -m housepred train")
    row = dict(EXAMPLE)
    for k in EXAMPLE:
        v = getattr(a, k, None)
        if v is not None:
            row[k] = v
    model = joblib.load(a.model)
    price = float(model.predict(pd.DataFrame([row]))[0])
    print("House description:")
    for k, v in row.items():
        print(f"  {k}: {v}")
    print(f"\nPredicted median house value for this area: ${price:,.0f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="housepred")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="run the whole pipeline and write results/")
    t.add_argument("--data", default=str(DEFAULT_DATA))
    t.add_argument("--results-dir", default=str(ROOT / "results"))
    t.add_argument("--models-dir", default=str(ROOT / "models"))
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--drop-capped", action="store_true", help="drop rows whose price is capped at $500,001")
    t.add_argument("--quick", action="store_true", help="smaller search, for a fast smoke test")
    t.set_defaults(func=cmd_train)
    q = sub.add_parser("predict", help="predict a price for one area (defaults to an example)")
    q.add_argument("--model", default=str(MODEL_PATH))
    for k, v in EXAMPLE.items():
        q.add_argument("--" + k.replace("_", "-"), dest=k, type=type(v), default=None, help=f"default {v}")
    q.set_defaults(func=cmd_predict)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
