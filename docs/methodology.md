# Methodology

## Observation and target

Each row is a message and its post-event book. The target is the sign of `mid[t+h] - mid[t]` at h ∈ {10, 50, 100}; absolute changes below 1e-9 USD are unchanged. The last h rows of each partition are excluded. The first 50 session rows provide feature history and are never training observations. No rows are shuffled. All models at a horizon use identical observations.

Training occupies raw rows [0, floor(.6N)); validation [.6N, floor(.8N)); test [.8N, N). Trailing features in validation/test may use earlier-partition history: that history was available at prediction time. No scaler, feature cutoff, or class prior fits future data.

## Features

| Feature | Definition |
|---|---|
| imbalance_l1 | (bid quantity − ask quantity) / total best-quote quantity |
| imbalance_l5 | Same ratio using first five levels (or available depth if fewer) |
| spread_bps | (ask − bid) / mid × 10,000 |
| depth_l1 | Sum of best bid and ask quantities |
| microprice_offset_bps | ((ask × bid quantity + bid × ask quantity) / depth − mid) / mid × 10,000 |
| ofi_20 | Sum of 20 best-quote order flow increments / current L1 depth |
| volatility_50 | Sample standard deviation of last 50 one-event mid returns × 10,000 |
| elapsed_20_ms | (time[t] − time[t−20]) × 1,000 |

The OFI increment is `1(bid[t] >= bid[t-1])*bq[t] - 1(bid[t] <= bid[t-1])*bq[t-1] - 1(ask[t] <= ask[t-1])*aq[t] + 1(ask[t] >= ask[t-1])*aq[t-1]`. This is quote-based OFI, not a reconstruction of individual orders. Event direction denotes the resting-order side, not the aggressor.

## Models and metrics

Class prior uses training label frequencies. Logistic models use `StandardScaler` then L2 logistic regression, C=1, max_iter=400. Expanded logistic uses all eight features. Boosting uses 100 iterations, max_leaf_nodes=15, learning_rate=.08, l2_regularization=1, random_state=42, and early_stopping=False. Other estimator defaults are recorded with package versions in each artifact.

Log loss uses all three labels. Multiclass Brier is the mean sum of the three squared probability errors (not divided by three). Balanced accuracy is mean recall over the three defined classes, assigning zero to a class with no support. Confusion rows are actual labels, columns predicted labels. Calibration groups upward probability into ten fixed intervals and reports count, mean prediction, and observed upward frequency. Sparse bins are unstable.

Liquidity and volatility tertiles come from training observations for each horizon. Test regimes can be highly unequal, and different class priors can explain apparent score differences. Conditional imbalance buckets describe the test set only; they are not parameter selection.

Paired loss differences versus the prior use a 200-replicate moving-block bootstrap, block size max(100, h). Positive difference favors the model. This is an exploratory interval conditional on one session and one block length, not a multiple-testing-adjusted significance claim.

## Execution accounting

Signals use unconditional three-class probabilities: buy when P(up) ≥ threshold, otherwise sell when P(down) ≥ threshold. Entry is at signal row + delay; exit at signal row + horizon, so delay consumes holding time. Delay ≥ horizon yields no eligible trades. Do not open another position until after the previous exit row. Quotes must remain within the test book; labels guarantee the exit boundary.

Longs enter at ask and exit at bid. Shorts enter at bid and cover at ask. Gross is `side × (exit − entry)`; fee is `(entry + exit) × fee_bps / 10,000` for one share. Net subtracts fees. Zero-delay same-event execution is optimistic. Exit delay is not modeled. Fees are a flat illustrative rate, not a venue schedule.

Drawdown includes initial zero equity and is measured at completed exits, not marked-to-market. No slippage beyond quote crossing, impact, queues, hidden liquidity, partial fills, borrow availability, or financing is modeled. Slider sweeps on test are exploratory, not out-of-sample optimization.

## Provenance

Historical exports contain input CSV SHA-256 hashes, package versions, full local Git HEAD and dirty status, UTC generation time, feature names, split protocol, and model parameters. Synthetic exports contain generator seed and an explicit synthetic provenance label. Artifacts contain all test-period book levels and probabilities, plus a decimated session chart. They contain no serialized executable model objects.
