"""Strict LOBSTER ingestion and explicitly synthetic integration fixtures."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]


def validate_frame(frame: pd.DataFrame) -> None:
    if len(frame) < 300:
        raise ValueError("At least 300 events required")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite input")
    if not frame.time.is_monotonic_increasing:
        raise ValueError("Messages must be chronological; equal timestamps retain source order")
    levels = len([c for c in frame if c.startswith("bid_price_")])
    if not levels:
        raise ValueError("No book levels")
    for level in range(1, levels + 1):
        bid, ask = frame[f"bid_price_{level}"], frame[f"ask_price_{level}"]
        if (bid <= 0).any() or (ask <= bid).any() or (ask.abs() >= 999999).any():
            raise ValueError("Nonpositive, locked, crossed, or unavailable book prices")
        for side in ("bid", "ask"):
            if (frame[f"{side}_size_{level}"] <= 0).any():
                raise ValueError("Book quantities must be positive")
        if level > 1:
            if (bid >= frame[f"bid_price_{level - 1}"]).any():
                raise ValueError("Bid levels must descend strictly")
            if (ask <= frame[f"ask_price_{level - 1}"]).any():
                raise ValueError("Ask levels must ascend strictly")
    if not frame.event_type.isin([1, 2, 3, 4, 5, 6, 7]).all():
        raise ValueError("Unknown LOBSTER event type")
    if not frame.direction.isin([-1, 1]).all():
        raise ValueError("Invalid resting-order direction")
    integer_cols = ["order_id", "size"] + [c for c in frame if "_size_" in c]
    if (frame[integer_cols] % 1 != 0).any().any():
        raise ValueError("Share counts and order IDs must be integers")
    if (frame["size"] < 0).any():
        raise ValueError("Negative event size")


def load_lobster(directory: str | Path) -> tuple[pd.DataFrame, dict]:
    directory = Path(directory)
    messages = sorted(directory.glob("*_message_*.csv"))
    books = sorted(directory.glob("*_orderbook_*.csv"))
    if len(messages) != 1 or len(books) != 1:
        raise ValueError("Provide exactly one paired session per directory")
    if messages[0].name.replace("_message_", "_orderbook_") != books[0].name:
        raise ValueError("Message and book filenames do not identify the same session")
    message = pd.read_csv(messages[0], header=None)
    book = pd.read_csv(books[0], header=None)
    if message.shape[1] != 6 or len(message) != len(book) or book.shape[1] % 4:
        raise ValueError("Invalid paired row counts or LOBSTER schema")
    message.columns = MESSAGE_COLUMNS
    book.columns = [
        f"{side}_{kind}_{level}"
        for level in range(1, book.shape[1] // 4 + 1)
        for side, kind in [("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size")]
    ]
    # Original prices are integer USD * 10,000. Books are post-event snapshots.
    for column in book:
        if "price" in column:
            book[column] = book[column] / 10000
    message["price"] = message["price"] / 10000
    frame = pd.concat([message, book], axis=1)
    validate_frame(frame)
    # A halt/crossing event needs session-specific segmentation, not silent filtering.
    if frame.event_type.isin([6, 7]).any():
        raise ValueError("Auction/halt events require session segmentation before this continuous-book study")
    return frame, {
        "kind": "historical",
        "session": messages[0].name.split("_message_")[0],
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [messages[0], books[0]]},
    }


def synthetic_book(n: int = 6000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """A deterministic plumbing fixture; never evidence of a market signal."""
    rng = np.random.default_rng(seed)
    # Price innovations are independent of displayed sizes by construction.
    mid = 100 + np.cumsum(rng.choice([-0.01, 0, 0.01], n, p=[0.3, 0.4, 0.3]))
    spread = rng.choice([0.02, 0.04, 0.06], n)
    frame = pd.DataFrame(
        {
            "time": 34200 + np.cumsum(rng.exponential(0.1, n)),
            "event_type": rng.choice([1, 2, 3, 4, 5], n),
            "order_id": np.arange(n),
            "size": rng.integers(1, 100, n),
            "price": mid,
            "direction": rng.choice([-1, 1], n),
        }
    )
    for level in range(1, 11):
        frame[f"bid_price_{level}"] = mid - spread / 2 - 0.01 * (level - 1)
        frame[f"ask_price_{level}"] = mid + spread / 2 + 0.01 * (level - 1)
        for side in ["bid", "ask"]:
            frame[f"{side}_size_{level}"] = rng.integers(1, 1000, n)
    validate_frame(frame)
    return frame, {
        "kind": "synthetic",
        "session": "SYNTHETIC — not AAPL or historical evidence",
        "seed": seed,
        "generator": "independent price innovations and random displayed depth",
    }
