"""Run with python -m orderbook.research --demo --output artifacts/research.json.gz."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import load_lobster, synthetic_book
from .features import FEATURES, labels, make_features, midprice, temporal_indices

CLASSES = [0, 1, 2]


def probability_matrix(model, x):
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 3))
    out[:, np.asarray(model.classes_, dtype=int)] = raw
    return out


def metrics(y, p):
    pred = p.argmax(axis=1)
    matrix = confusion_matrix(y, pred, labels=CLASSES)
    counts = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), counts, out=np.zeros(3, dtype=float), where=counts > 0)
    return {
        "n": len(y),
        "log_loss": float(log_loss(y, p, labels=CLASSES)),
        "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(recall.mean()),
        "confusion": matrix.tolist(),
        "class_counts": np.bincount(y, minlength=3).tolist(),
    }


def calibration(y, p):
    bins = np.minimum((p[:, 2] * 10).astype(int), 9)
    return [
        {
            "bin": k,
            "n": int((bins == k).sum()),
            "predicted": float(p[bins == k, 2].mean()),
            "observed": float((y[bins == k] == 2).mean()),
        }
        for k in range(10)
        if (bins == k).any()
    ]


def block_loss_interval(y, p, prior, block=100, seed=42):
    """Exploratory paired moving-block bootstrap of per-event log-loss improvement."""
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    base = -np.log(np.clip(prior[np.arange(len(y)), y], 1e-15, 1))
    delta = base - loss
    block = min(block, len(y))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(200):
        starts = rng.integers(0, len(y) - block + 1, int(np.ceil(len(y) / block)))
        ix = (starts[:, None] + np.arange(block)).ravel()[: len(y)]
        estimates.append(delta[ix].mean())
    return {
        "mean": float(delta.mean()),
        "interval_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "block_events": block,
        "replicates": 200,
        "interpretation": "exploratory; single-session dependence may exceed block length",
    }


def run(book, provenance, horizons=(10, 50, 100)):
    x = make_features(book)
    mid = midprice(book)
    test_start = int(0.8 * len(book))
    results = {}
    with threadpool_limits(limits=2):
        for h in horizons:
            train, valid, test = temporal_indices(len(book), h)
            y = labels(book, h).to_numpy()
            definitions = {
                "Class prior": (DummyClassifier(strategy="prior"), FEATURES),
                "Queue imbalance": (
                    make_pipeline(StandardScaler(), LogisticRegression(max_iter=400)),
                    ["imbalance_l1"],
                ),
                "Logistic expanded": (
                    make_pipeline(StandardScaler(), LogisticRegression(max_iter=400)),
                    FEATURES,
                ),
                "Boosted trees": (
                    HistGradientBoostingClassifier(
                        max_iter=100,
                        max_leaf_nodes=15,
                        learning_rate=0.08,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=42,
                    ),
                    FEATURES,
                ),
            }
            models = {}
            for name, (model, columns) in definitions.items():
                model.fit(x.iloc[train][columns], y[train])
                pv = probability_matrix(model, x.iloc[valid][columns])
                pt = probability_matrix(model, x.iloc[test][columns])
                regimes = {}
                for feature in ["depth_l1", "volatility_50"]:
                    cutoffs = np.quantile(x.iloc[train][feature], [1 / 3, 2 / 3])
                    groups = np.searchsorted(cutoffs, x.iloc[test][feature], side="right")
                    regimes[feature] = {
                        "training_cutoffs": cutoffs.tolist(),
                        "groups": [
                            {"group": label, **metrics(y[test][groups == k], pt[groups == k])}
                            for k, label in enumerate(["Low", "Middle", "High"])
                            if (groups == k).any()
                        ],
                    }
                models[name] = {
                    "validation": metrics(y[valid], pv),
                    "test": metrics(y[test], pt),
                    "probabilities": pt.tolist(),
                    "calibration": calibration(y[test], pt),
                    "regimes": regimes,
                    "parameters": {k: str(v) for k, v in model.get_params().items()},
                }
            prior = np.array(models["Class prior"]["probabilities"])
            for model_result in models.values():
                model_result["loss_improvement"] = block_loss_interval(
                    y[test], np.array(model_result["probabilities"]), prior, max(100, h)
                )
            # Conditional probabilities computed on test, not used to choose signals.
            bins = np.clip(((x.iloc[test].imbalance_l1.to_numpy() + 1) * 5).astype(int), 0, 9)
            conditional = [
                {
                    "imbalance_center": -0.9 + 0.2 * k,
                    "n": int((bins == k).sum()),
                    "class_probabilities": (
                        np.bincount(y[test][bins == k], minlength=3) / (bins == k).sum()
                    ).tolist(),
                }
                for k in range(10)
                if (bins == k).any()
            ]
            results[str(h)] = {
                "train_n": len(train),
                "validation_n": len(valid),
                "test_n": len(test),
                "test_indices": (test - test_start).tolist(),
                "test_labels": y[test].tolist(),
                "models": models,
                "conditional_probabilities": conditional,
            }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    chart_idx = np.unique(np.linspace(0, len(book) - 1, min(2000, len(book)), dtype=int))
    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance,
        "code": {
            "commit": commit,
            "dirty": dirty,
            "source_sha256": {
                str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        },
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn"]},
        "protocol": {
            "classes": ["Down", "Unchanged", "Up"],
            "split": [0.6, 0.2, 0.2],
            "warmup": 50,
            "features": FEATURES,
            "seed": 42,
            "horizons": list(horizons),
            "label": "sign(mid[t+h]-mid[t]); absolute tolerance 1e-9 USD",
            "test_start_row": test_start,
        },
        "sample": {
            "rows": len(book),
            "levels": len([c for c in book if c.startswith("bid_price_")]),
            "mean_spread_bps": float(x.spread_bps.mean()),
            "chart": {"time": book.iloc[chart_idx].time.tolist(), "mid": mid.iloc[chart_idx].tolist()},
        },
        "test_book": {c: book.iloc[test_start:][c].tolist() for c in book},
        "results": results,
    }


def save_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf8") as f:
        json.dump(artifact, f, allow_nan=False, separators=(",", ":"))


def load_artifact(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--data", type=Path)
    parser.add_argument("--output", default="artifacts/research.json.gz")
    parser.add_argument("--events", type=int, default=6000, help="Synthetic demo events only")
    args = parser.parse_args()
    book, provenance = synthetic_book(args.events) if args.demo else load_lobster(args.data)
    artifact = run(book, provenance)
    save_artifact(artifact, args.output)
    print(f"Wrote {args.output}: {len(book):,} {provenance['kind']} events")
    for h, result in artifact["results"].items():
        print(h, {name: round(model["test"]["log_loss"], 4) for name, model in result["models"].items()})


if __name__ == "__main__":
    main()
