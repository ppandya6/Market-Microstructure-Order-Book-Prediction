"""Run: streamlit run app.py. All research values are read from pipeline artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from orderbook.execution import simulate, summary
from orderbook.research import load_artifact

st.set_page_config(page_title="Microstructure Research Lab", page_icon="μ", layout="wide")
st.markdown(
    """<style>
.stApp {background:#f4f6f8;color:#1c3036} [data-testid="stSidebar"]{background:#153034}
[data-testid="stSidebar"] * {color:#c8ded8}
[data-testid="stSidebar"] input {color:#1c3036}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:10px;border-radius:6px}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#2b484c} h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:white;border:1px solid #dfe5e7;border-radius:8px;padding:20px}
[data-testid="stMetricValue"]{font-family:monospace;font-size:26px} .eyebrow{letter-spacing:2px;color:#71888b;font-size:11px}
.stButton>button,.stDownloadButton>button {border-color:#dbe3e4} section.main>div{max-width:1600px}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_data
def read(path, modified):
    return load_artifact(path)


def line(x, ys, labels, ytitle="", cursor=None):
    if len(x) and isinstance(x[0], str) and ":" in x[0]:
        x = pd.to_datetime(x, format="%H:%M:%S.%f")
    fig = go.Figure()
    colors = ["#368e79", "#719cc4", "#ba7781"]
    for y, label, color in zip(ys, labels, colors):
        fig.add_trace(go.Scatter(x=x, y=y, name=label, line={"color": color, "width": 2}))
    if cursor is not None:
        fig.add_vline(x=cursor, line_dash="dot", line_color="#ba7781")
    fig.update_layout(
        template="plotly_white",
        height=330,
        margin={"l": 15, "r": 15, "t": 20, "b": 20},
        yaxis_title=ytitle,
        legend={"orientation": "h"},
        paper_bgcolor="white",
        xaxis={"showgrid": False, "nticks": 6, "tickangle": 0},
        yaxis={"gridcolor": "#edf0f0"},
    )
    st.plotly_chart(fig, width="stretch")


def card(label, value, help=None):
    st.metric(label, value, help=help)


def clock(seconds):
    return pd.to_datetime(seconds, unit="s").strftime("%H:%M:%S.%f")[:-3]


with st.sidebar:
    st.markdown("## μ microstructure")
    st.caption("R E S E A R C H   L A B")
    st.divider()
    page = st.radio(
        "WORKSPACE", ["Research desk", "Book replay", "Experiments", "Execution", "Method & data"]
    )
    st.divider()
    st.markdown("Does displayed liquidity anticipate the next price move?")
    st.caption("Prediction → execution")
    artifact_path = st.text_input("Research artifact", "artifacts/research.json.gz")
    st.caption("Local file produced by the Python pipeline. No live trading connections.")

path = Path(artifact_path)
if not path.is_file():
    st.title("Prepare your research artifact")
    st.info("Run the demo or supply a paired historical session, then refresh this page.")
    st.code(
        "python -m orderbook.research --demo --output artifacts/research.json.gz\n# Or: python -m orderbook.research --data data/raw/aapl-2012-06-21"
    )
    st.stop()
try:
    a = read(str(path), path.stat().st_mtime_ns)
    if a["schema_version"] != 1:
        raise ValueError("Unsupported artifact schema")
    book = pd.DataFrame(a["test_book"])
except (ValueError, OSError, KeyError) as error:
    st.error(f"Cannot load artifact: {error}")
    st.stop()

st.markdown('<p class="eyebrow">MARKET MICROSTRUCTURE / FINANCIAL ML</p>', unsafe_allow_html=True)
if a["provenance"]["kind"] == "synthetic":
    st.warning(
        "SYNTHETIC DEMO · Randomly generated plumbing fixture. These are not historical AAPL results or evidence of predictability."
    )
else:
    st.info("Exploratory historical sample · " + a["provenance"]["session"])


def selection():
    left, right = st.columns(2)
    h = left.selectbox(
        "Prediction horizon · subsequent events",
        list(a["results"]),
        index=list(a["results"]).index("50") if "50" in a["results"] else 0,
    )
    result = a["results"][h]
    model_name = right.selectbox("Model", list(result["models"]), index=2)
    return int(h), result, model_name, result["models"][model_name]


if page == "Research desk":
    st.title("A closer look at the book.")
    st.caption("From displayed liquidity to price prediction and execution.")
    st.info(
        "First research checkpoint · Four model specifications, three event horizons. Additional reserved dates are required for conclusions about persistence."
    )
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Paired events", "Displayed depth", "Mean quoted spread", "Data source"],
        [
            f"{a['sample']['rows']:,}",
            str(a["sample"]["levels"]) + " levels",
            f"{a['sample']['mean_spread_bps']:.2f} bps",
            a["provenance"]["kind"].upper(),
        ],
    ):
        with col:
            card(label, value)
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Mid-price through the session")
        chart = a["sample"]["chart"]
        line([clock(t) for t in chart["time"]], [chart["mid"]], ["Mid-price"], "USD")
        st.caption(
            "Chronological partitions: first 60% training, next 20% validation, final 20% test. Chart is decimated for display."
        )
    with right:
        st.subheader("Research protocol")
        st.markdown(
            "**Three-class price prediction**\n\nDown, unchanged, or up after 10, 50, or 100 subsequent messages.\n\n**Causal book and flow inputs**\n\nPost-event quotes and trailing observations.\n\n**Temporal evaluation**\n\nTargets crossing partition boundaries are purged."
        )
    st.subheader("Baseline to nonlinear prediction")
    rows = [
        {
            "Horizon": h,
            "Model": name,
            "Validation log loss": m["validation"]["log_loss"],
            "Test log loss": m["test"]["log_loss"],
        }
        for h, r in a["results"].items()
        for name, m in r["models"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Research journal")
    note = st.text_area("Record hypotheses before inspecting another result")
    if st.button("Add note") and note.strip():
        st.session_state.setdefault("journal", []).append(
            {"time_utc": pd.Timestamp.now(tz="UTC").isoformat(), "note": note}
        )
    notes = st.session_state.get("journal", [])
    st.caption("Notes live in this browser session. Download to keep them.")
    if notes:
        st.dataframe(notes, hide_index=True)
        st.download_button("Download journal", json.dumps(notes, indent=2), "journal.json")

elif page == "Book replay":

    @st.fragment(run_every=0.4 if st.session_state.get("playing", False) else None)
    def replay_view():
        st.title("Inside the order book.")
        st.caption(
            "A contiguous window from the beginning of the test partition. Post-event snapshots; source order retained."
        )
        limit = min(1500, len(book))
        if "replay" not in st.session_state:
            st.session_state.replay = 0
        if st.button("Pause" if st.session_state.get("playing", False) else "▶ Play"):
            st.session_state.playing = not st.session_state.get("playing", False)
            st.rerun()
        if st.session_state.get("playing", False):
            st.session_state.replay = (st.session_state.replay + 1) % limit
        c1, c2 = st.columns([1, 7])
        if c1.button("+1 event"):
            st.session_state.replay = (st.session_state.replay + 1) % limit
        i = c2.slider("Event", 0, limit - 1, key="replay")
        row = book.iloc[i]
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Displayed liquidity")
            levels = a["sample"]["levels"]
            ladder = pd.DataFrame(
                {
                    "Bid price / USD": [row[f"bid_price_{j}"] for j in range(1, levels + 1)],
                    "Bid shares": [row[f"bid_size_{j}"] for j in range(1, levels + 1)],
                    "Ask shares": [row[f"ask_size_{j}"] for j in range(1, levels + 1)],
                    "Ask price / USD": [row[f"ask_price_{j}"] for j in range(1, levels + 1)],
                }
            )
            st.dataframe(
                ladder.style.bar(subset=["Bid shares"], color="#c3dfd6")
                .bar(subset=["Ask shares"], color="#efd2d7")
                .format(precision=2),
                hide_index=True,
                height=400,
                width="stretch",
            )
            cols = st.columns(3)
            with cols[0]:
                card("Mid-price", f"${(row.bid_price_1 + row.ask_price_1) / 2:.3f}")
            with cols[1]:
                card("Spread", f"${row.ask_price_1 - row.bid_price_1:.3f}")
            with cols[2]:
                card(
                    "L1 imbalance",
                    f"{(row.bid_size_1 - row.ask_size_1) / (row.bid_size_1 + row.ask_size_1):.3f}",
                )
        with right:
            st.subheader("Current event")
            event_names = {
                1: "Order submitted",
                2: "Partial cancellation",
                3: "Order deleted",
                4: "Visible execution",
                5: "Hidden execution",
            }
            st.write(event_names.get(int(row.event_type), "Other event"))
            st.json(
                {
                    "time (session clock)": clock(row.time),
                    "source row (1-based)": a["protocol"]["test_start_row"] + i + 1,
                    "resting side": "Buy limit" if row.direction == 1 else "Sell limit",
                    "size": int(row["size"]),
                    "price": float(row.price),
                    "order_id": int(row.order_id),
                }
            )
            st.caption(
                "Execution direction is the resting order side; a sell-limit execution is buyer-initiated."
            )
            st.subheader("Window mid-price")
            line(
                np.arange(limit),
                [((book.bid_price_1 + book.ask_price_1) / 2).iloc[:limit]],
                ["Mid-price"],
                "USD",
                i,
            )

    replay_view()

elif page == "Experiments":
    st.title("From imbalance to prediction.")
    h, r, name, model = selection()
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Test observations", "Test log loss", "Balanced accuracy", "Multiclass Brier"],
        [
            f"{model['test']['n']:,}",
            f"{model['test']['log_loss']:.4f}",
            f"{model['test']['balanced_accuracy']:.1%}",
            f"{model['test']['brier']:.4f}",
        ],
    ):
        with col:
            card(label, value)
    st.subheader("Model comparison")
    table = pd.DataFrame(
        [
            {
                "Model": n,
                "Validation log loss": m["validation"]["log_loss"],
                "Test log loss": m["test"]["log_loss"],
                "Test Brier": m["test"]["brier"],
                "Accuracy": m["test"]["accuracy"],
                "Balanced accuracy": m["test"]["balanced_accuracy"],
            }
            for n, m in r["models"].items()
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Export metrics", table.to_csv(index=False), f"metrics-h{h}.csv")
    left, right = st.columns(2)
    with left:
        st.subheader("Upward-move probability calibration")
        cal = model["calibration"]
        line(
            [c["predicted"] for c in cal],
            [[c["observed"] for c in cal], [c["predicted"] for c in cal]],
            ["Observed", "Ideal"],
            "Observed upward frequency",
        )
        st.dataframe(pd.DataFrame(cal), hide_index=True)
    with right:
        st.subheader("Performance across conditions")
        for feature, groups in model["regimes"].items():
            st.caption(feature + " · cutoffs from training observations")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Condition": g["group"], "n": g["n"], "Log loss": g["log_loss"]}
                        for g in groups["groups"]
                    ]
                ),
                hide_index=True,
            )
        st.subheader("Paired loss improvement over class prior")
        st.json(model["loss_improvement"])
    st.subheader("Conditional move probabilities · held-out test")
    conditional = r["conditional_probabilities"]
    line(
        [c["imbalance_center"] for c in conditional],
        [[c["class_probabilities"][j] for c in conditional] for j in range(3)],
        ["Down", "Unchanged", "Up"],
        "Probability",
    )
    st.caption("Exploratory associations; imbalance buckets have unequal counts. No causal claim.")
    st.subheader("Confusion matrix · rows actual, columns predicted")
    st.dataframe(
        pd.DataFrame(
            model["test"]["confusion"], index=["Down", "Unchanged", "Up"], columns=["Down", "Unchanged", "Up"]
        )
    )
    st.warning(
        "Repeated inspection makes this session exploratory. Reserve additional dates before confirmatory evaluation."
    )

elif page == "Execution":
    st.title("What survives the spread?")
    st.caption("Aggressive one-share round trips from test-period predictions. One position at a time.")
    h, r, name, model = selection()
    cols = st.columns(3)
    fee = cols[0].selectbox("Fee · bps each side", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0], index=3)
    delay = cols[1].selectbox("Order delay · events", [0, 1, 5, 10, 25], index=3)
    threshold = cols[2].selectbox("Direction probability", [0.5, 0.55, 0.6, 0.7, 0.8, 0.9], index=2)
    p = np.array(model["probabilities"])
    trades = simulate(book, r["test_indices"], p, h, delay, fee, threshold)
    s = summary(trades)
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Net P&L", "Completed round trips", "Win rate", "Trade-end drawdown"],
        [
            f"${s['net']:.2f}",
            s["trades"],
            "—" if s["win_rate"] is None else f"{s['win_rate']:.1%}",
            f"${s['drawdown']:.2f}",
        ],
    ):
        with col:
            card(label, value)
    if len(trades):
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Cumulative trading P&L")
            line(
                np.arange(len(trades) + 1),
                [np.r_[0, trades.net.cumsum()], np.r_[0, trades.gross.cumsum()]],
                ["Net after fees", "Gross after spread"],
                "USD",
            )
        with right:
            st.subheader("Fee sensitivity")
            fees = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
            nets = [
                float((trades.gross - (trades.entry_price + trades.exit_price) * f / 10000).sum())
                for f in fees
            ]
            line(fees, [nets], ["Net P&L"], "USD")
        st.subheader("Trade ledger")
        st.dataframe(trades.tail(12), hide_index=True, width="stretch")
    else:
        st.info(
            "No eligible completed trades at these settings. Delay must be shorter than the prediction horizon."
        )
    st.download_button("Export all trades", trades.to_csv(index=False), "trades.csv")
    st.warning(
        "Exploratory controls use the test period. They must not be treated as out-of-sample parameter selection."
    )
    st.caption(
        "Buys cross the ask; sells cross the bid. Exit at signal time + horizon; entry at signal time + delay. No impact, queue position, borrow constraints, partial fills, or exit latency. Same-event entry at zero delay is optimistic. Drawdown is measured at completed exits only."
    )

else:
    st.title("Inspect every assumption.")
    st.caption("A Python research pipeline with an interactive inspection layer.")
    left, right = st.columns(2)
    with left:
        st.subheader("Data provenance")
        st.json(a["provenance"])
        st.subheader("Target and feature definitions")
        st.write(a["protocol"])
        st.markdown(
            "L1 imbalance = (bid quantity − ask quantity) / total quantity. L5 uses summed quantities over up to five levels. OFI aggregates best-quote additions/removals over 20 events, divided by current L1 depth. Volatility uses the last 50 one-event midpoint returns. Elapsed time spans 20 events. Microprice weights each quote by opposite-side quantity."
        )
    with right:
        st.subheader("Chronological partitions")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Horizon": h,
                        "Train": r["train_n"],
                        "Validation": r["validation_n"],
                        "Test": r["test_n"],
                    }
                    for h, r in a["results"].items()
                ]
            ),
            hide_index=True,
        )
        st.markdown(
            "First 50 rows supply feature history. Any label reaching the next partition is excluded. Scalers and regime tertiles fit training rows only. Equal timestamps preserve message order. Windows may use earlier-partition history because it was available in real time."
        )
        st.subheader("Model specifications")
        st.markdown(
            "Training class prior; standardized L1 logistic regression; eight-feature standardized logistic regression; histogram gradient boosting with 100 iterations, 15 leaves, learning rate 0.08, L2=1. Internal random early stopping is disabled. Hyperparameters are fixed, not selected on test."
        )
    st.subheader("Run the Python pipeline")
    st.code(
        "python -m pip install -e '.[dev]'\npython scripts/download_sample.py\npython -m orderbook.research --data data/raw/aapl-2012-06-21\npython -m pytest\nstreamlit run app.py"
    )
    st.download_button(
        "Export provenance and protocol",
        json.dumps(
            {k: a[k] for k in ["provenance", "protocol", "code", "packages", "created_utc"]}, indent=2
        ),
        "provenance.json",
    )
    st.download_button(
        "Download research artifact",
        path.read_bytes(),
        path.name,
        mime="application/gzip" if path.suffix == ".gz" else "application/json",
    )
    st.subheader("Reproducibility")
    st.json({k: a[k] for k in ["created_utc", "code", "python", "packages"]})
    st.subheader("Next research phase")
    st.write(
        "Additional dates and symbols; rolling-origin evaluation; training/validation ablations; alternative bootstrap block lengths; timestamp-based latency; reserved confirmatory periods. Passive execution requires a separate queue and fill model."
    )
