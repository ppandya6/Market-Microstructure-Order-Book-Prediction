"""Causal, post-event features; labels are built separately."""

import numpy as np
import pandas as pd

FEATURES = [
    "imbalance_l1",
    "imbalance_l5",
    "spread_bps",
    "depth_l1",
    "microprice_offset_bps",
    "ofi_20",
    "volatility_50",
    "elapsed_20_ms",
]
WARMUP = 50


def midprice(book):
    return (book.bid_price_1 + book.ask_price_1) / 2


def make_features(book: pd.DataFrame) -> pd.DataFrame:
    bid, ask = book.bid_price_1, book.ask_price_1
    bq, aq = book.bid_size_1, book.ask_size_1
    depth, mid = bq + aq, midprice(book)
    levels = min(5, len([c for c in book if c.startswith("bid_size_")]))
    b5 = book[[f"bid_size_{i}" for i in range(1, levels + 1)]].sum(axis=1)
    a5 = book[[f"ask_size_{i}" for i in range(1, levels + 1)]].sum(axis=1)
    ofi = (
        (bid >= bid.shift()).astype(int) * bq
        - (bid <= bid.shift()).astype(int) * bq.shift()
        - (ask <= ask.shift()).astype(int) * aq
        + (ask >= ask.shift()).astype(int) * aq.shift()
    )
    features = pd.DataFrame(
        {
            "imbalance_l1": (bq - aq) / depth,
            "imbalance_l5": (b5 - a5) / (b5 + a5),
            "spread_bps": (ask - bid) / mid * 10000,
            "depth_l1": depth,
            "microprice_offset_bps": ((ask * bq + bid * aq) / depth - mid) / mid * 10000,
            "ofi_20": ofi.rolling(20).sum() / depth,
            "volatility_50": mid.pct_change().rolling(50).std() * 10000,
            "elapsed_20_ms": book.time.diff(20) * 1000,
        }
    )
    return features[FEATURES]


def labels(book, horizon):
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    delta = midprice(book).shift(-horizon) - midprice(book)
    # Tolerance handles decimal floating arithmetic; unchanged remains a class.
    return pd.Series(np.where(delta.abs() < 1e-9, 1, np.where(delta > 0, 2, 0)), index=book.index)


def temporal_indices(n, horizon):
    """60/20/20 raw boundaries; remove targets reaching the next partition."""
    a, b = int(0.6 * n), int(0.8 * n)
    parts = [np.arange(WARMUP, a - horizon), np.arange(a, b - horizon), np.arange(b, n - horizon)]
    if any(len(p) < 20 for p in parts):
        raise ValueError("Insufficient observations for this horizon")
    return parts
