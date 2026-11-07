# Research readout

Source: **synthetic** — SYNTHETIC — not AAPL or historical evidence
Generated: 2026-11-06T23:51:14.331827+00:00

Exploratory results; synthetic data is not market evidence.

| Horizon | Model | Validation log loss | Test log loss | Balanced accuracy | Net USD* | Trades* |
|---|---|---|---|---|---|---|
| 10 | Class prior | 1.0188 | 1.0375 | 33.3% | 0.00 | 0 |
| 10 | Queue imbalance | 1.0188 | 1.0375 | 33.3% | 0.00 | 0 |
| 10 | Logistic expanded | 1.0257 | 1.0587 | 30.9% | -0.06 | 1 |
| 10 | Boosted trees | 1.1102 | 1.1275 | 34.4% | -4.14 | 77 |
| 50 | Class prior | 0.8719 | 0.8098 | 33.3% | 0.00 | 0 |
| 50 | Queue imbalance | 0.8719 | 0.8098 | 33.3% | 0.00 | 0 |
| 50 | Logistic expanded | 0.8804 | 0.8348 | 32.0% | -0.23 | 5 |
| 50 | Boosted trees | 1.0702 | 0.9544 | 30.9% | -1.00 | 22 |
| 100 | Class prior | 0.8002 | 0.7793 | 33.3% | 0.00 | 0 |
| 100 | Queue imbalance | 0.8007 | 0.7795 | 33.3% | 0.00 | 0 |
| 100 | Logistic expanded | 0.8171 | 0.8278 | 30.8% | -0.68 | 6 |
| 100 | Boosted trees | 1.1613 | 1.3020 | 33.6% | -0.91 | 11 |

*Execution: 1 event delay, 0.5 bps fee each side, 60% directional threshold, one share, nonoverlapping trades. These settings are illustrative, not selected for performance.

Reserve additional dates for confirmatory evaluation. Quote-based execution omits market impact, borrow constraints, and passive fills.
