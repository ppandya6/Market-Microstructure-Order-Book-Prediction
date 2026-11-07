# Implementation status

This is a new implementation from the supplied screenshots, not the original site's downloaded source. The starting repository contained only a title README.

## Implemented

- Installable Python package and a five-view Streamlit interface with green/teal styling.
- Strict paired LOBSTER ingestion, USD price conversion, validation, file hashing, and documented public sample downloader.
- Seeded synthetic integration fixture with explicit provenance and no invented historical results.
- Eight causal features, three targets, purged chronological splits, and four fixed model specifications.
- Validation/test metrics, calibration counts, confusion matrices, conditional probabilities, training-defined liquidity/volatility groups, and exploratory paired block-bootstrap intervals.
- Quote-crossing execution with delays, fees, one-position constraint, ledger, sensitivity curve, and trade-end drawdown.
- Replay slider, step, and play/pause controls; metric/trade/provenance downloads; session-scoped research journal export.
- JSON/gzip artifact CLI and Markdown readout exporter; package-version snapshot; CI.

## Verified in the current environment

- Editable package installation in an isolated Python 3.12 environment.
- A second clean `.venv` installation from `requirements-tested.txt` using `bash scripts/setup.sh`; all 19 tests and lint passed there too, and `pip check` reported no broken requirements.
- Reusable `install_script` and `start_skill` saved in the environment draft (not published).
- Full synthetic 6,000-event experiment: all four models at horizons 10, 50, 100.
- 19 automated tests passed: research invariants, paired CSV ingestion, end-to-end artifact generation, and all five Streamlit pages.
- Ruff checks passed.
- Streamlit process started and health endpoint returned `ok`.
- Chromium browser smoke check: all five views rendered without exception panels; replay play/pause worked.

## Outstanding external dependency

The environment proxy returns CONNECT 403 for the original research site and `lobsterdata.com`. Domain additions have been saved to the environment configuration draft; they are not yet applied/published. Historical sample download and the screenshots' AAPL statistics remain unverified. The downloader's remote URL must be confirmed once network access is applied. Historical ingestion itself is covered by CSV round-trip tests with generated fixtures.

## Research limitations

No claims about profitability, prediction persistence, or cross-day generalization are established. Multi-session walk-forward evaluation, reserved confirmatory dates, ablations, timestamp latency, market impact, passive queue fills, and borrow constraints remain further research. The bootstrap is a single block-length exploratory diagnostic. App controls on test data do not provide valid parameter selection.

## Delivery state

Source files are prepared in the local repository. No GitHub push or external deployment is performed by the app or research commands. CI is provided but a remote GitHub Actions run is not claimed. A fresh cloud snapshot restoration has not been tested.
