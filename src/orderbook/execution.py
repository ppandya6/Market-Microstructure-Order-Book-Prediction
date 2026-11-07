"""Aggressive one-share quote execution, nonoverlapping positions, no fill model."""

import numpy as np
import pandas as pd

TRADE_COLUMNS = [
    "signal",
    "entry",
    "exit",
    "time",
    "side",
    "entry_price",
    "exit_price",
    "gross",
    "fees",
    "net",
]


def simulate(book, indices, probabilities, horizon, delay=0, fee_bps=0.5, threshold=0.6):
    if horizon < 1 or delay < 0 or fee_bps < 0 or not 0.5 <= threshold <= 1:
        raise ValueError("Invalid execution settings")
    indices, probabilities = np.asarray(indices), np.asarray(probabilities)
    if probabilities.shape != (len(indices), 3) or not np.isfinite(probabilities).all():
        raise ValueError("Expected aligned finite three-class probabilities")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid probability simplex")
    if len(indices) and ((np.diff(indices) <= 0).any() or indices.min() < 0 or indices.max() >= len(book)):
        raise ValueError("Signal indices must be in-bounds and strictly increasing")
    # Holding horizon is measured from signal time, including order delay.
    if delay >= horizon:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    rows, busy_until = [], -1
    for i, p in zip(indices, probabilities):
        entry, exit_ = int(i + delay), int(i + horizon)
        if i <= busy_until or exit_ >= len(book):
            continue
        side = 1 if p[2] >= threshold else -1 if p[0] >= threshold else 0
        if side == 0:
            continue
        buy = "ask_price_1" if side == 1 else "bid_price_1"
        sell = "bid_price_1" if side == 1 else "ask_price_1"
        entry_price, exit_price = float(book.iloc[entry][buy]), float(book.iloc[exit_][sell])
        gross = side * (exit_price - entry_price)
        fees = (entry_price + exit_price) * fee_bps / 10000
        rows.append(
            [
                int(i),
                entry,
                exit_,
                float(book.iloc[exit_].time),
                "long" if side == 1 else "short",
                entry_price,
                exit_price,
                gross,
                fees,
                gross - fees,
            ]
        )
        busy_until = exit_
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def summary(trades):
    equity = np.r_[0.0, trades.net.to_numpy(dtype=float).cumsum()]
    return {
        "trades": len(trades),
        "net": float(equity[-1]),
        "win_rate": float((trades.net > 0).mean()) if len(trades) else None,
        "drawdown": float((np.maximum.accumulate(equity) - equity).max()),
    }
