import numpy as np
import pandas as pd
import pytest

from orderbook.data import load_lobster, synthetic_book, validate_frame
from orderbook.execution import simulate, summary
from orderbook.features import labels, make_features, temporal_indices
from orderbook.research import load_artifact, run, save_artifact


def test_feature_prefix_invariance():
    book, _ = synthetic_book(1000)
    full = make_features(book)
    pd.testing.assert_frame_equal(full.iloc[:700], make_features(book.iloc[:700]))
    assert np.isfinite(full.iloc[50:]).all().all()


def test_future_book_cannot_change_current_feature():
    book, _ = synthetic_book(1000)
    before = make_features(book)
    book.loc[701:, "bid_size_1"] *= 50
    pd.testing.assert_frame_equal(before.iloc[:701], make_features(book).iloc[:701])


@pytest.mark.parametrize("h", [10, 50, 100])
def test_purged_partitions(h):
    train, valid, test = temporal_indices(6000, h)
    assert train[0] == 50
    assert train[-1] + h < 3600
    assert valid[0] == 3600 and valid[-1] + h < 4800
    assert test[0] == 4800 and test[-1] + h < 6000


def test_three_way_labels():
    book, _ = synthetic_book(500)
    book.loc[:3, "bid_price_1"] = [99, 100, 100, 99]
    book.loc[:3, "ask_price_1"] = [101, 102, 102, 101]
    assert labels(book, 1).iloc[:3].tolist() == [2, 1, 0]


def test_strict_input_checks():
    book, _ = synthetic_book(500)
    book.loc[5, "time"] = -1
    with pytest.raises(ValueError, match="chronological"):
        validate_frame(book)
    book, _ = synthetic_book(500)
    book.loc[5, "ask_price_1"] = book.loc[5, "bid_price_1"]
    with pytest.raises(ValueError, match="crossed"):
        validate_frame(book)


def flat_book():
    return pd.DataFrame(
        {"time": np.arange(20), "bid_price_1": np.full(20, 99.0), "ask_price_1": np.full(20, 101.0)}
    )


@pytest.mark.parametrize("p", [[0, 0, 1], [1, 0, 0]])
def test_crossing_and_two_sided_fees(p):
    trades = simulate(flat_book(), [0], [p], 5, 1, 1, 0.6)
    assert trades.iloc[0]["gross"] == -2
    assert trades.iloc[0]["fees"] == pytest.approx(0.02)
    assert trades.iloc[0]["net"] == pytest.approx(-2.02)
    assert summary(trades)["drawdown"] == pytest.approx(2.02)


def test_nonoverlap_and_delay():
    book = flat_book()
    p = np.tile([0, 0, 1], (14, 1))
    trades = simulate(book, np.arange(14), p, 5, 2, 0, 0.6)
    assert trades.signal.tolist() == [0, 6, 12]
    assert trades.entry.tolist() == [2, 8, 14]
    assert len(simulate(book, [0], [[0, 0, 1]], 5, 5)) == 0
    assert len(simulate(book, [0], [[0.2, 0.6, 0.2]], 5)) == 0


def test_exit_prices_and_horizon():
    book = flat_book()
    book.loc[5, "bid_price_1"] = 105
    trade = simulate(book, [0], [[0, 0, 1]], 5, 2, 0).iloc[0]
    assert trade["entry"] == 2 and trade["exit"] == 5 and trade["gross"] == 4


def test_probability_validation():
    with pytest.raises(ValueError, match="simplex"):
        simulate(flat_book(), [0], [[0.5, 0.5, 0.5]], 5)


def test_real_csv_ingestion(tmp_path):
    book, _ = synthetic_book(500)
    message = book[["time", "event_type", "order_id", "size", "price", "direction"]].copy()
    message["price"] = (message.price * 10000).round().astype(int)
    message.to_csv(tmp_path / "TEST_message_10.csv", index=False, header=False)
    columns = [
        f"{side}_{kind}_{i}"
        for i in range(1, 11)
        for side, kind in [("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size")]
    ]
    raw = book[columns].copy()
    for c in raw:
        if "price" in c:
            raw[c] = (raw[c] * 10000).round().astype(int)
    raw.to_csv(tmp_path / "TEST_orderbook_10.csv", index=False, header=False)
    loaded, provenance = load_lobster(tmp_path)
    assert len(loaded) == 500 and len(provenance["files"]) == 2
    np.testing.assert_allclose(loaded.bid_price_1, book.bid_price_1)


def test_end_to_end_artifact(tmp_path):
    book, provenance = synthetic_book(1000)
    result = run(book, provenance, horizons=(10,))
    models = result["results"]["10"]["models"]
    assert len(models) == 4
    for model in models.values():
        p = np.array(model["probabilities"])
        np.testing.assert_allclose(p.sum(axis=1), 1)
        assert model["test"]["log_loss"] > 0
        assert sum(model["test"]["class_counts"]) == model["test"]["n"]
    path = tmp_path / "research.json.gz"
    save_artifact(result, path)
    assert load_artifact(path)["provenance"] == provenance
    test_book = pd.DataFrame(result["test_book"])
    r = result["results"]["10"]
    trades = simulate(test_book, r["test_indices"], models["Boosted trees"]["probabilities"], 10, 1, 0.5, 0.5)
    assert (trades["exit"] < len(test_book)).all()
