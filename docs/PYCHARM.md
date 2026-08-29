# PyCharm pipeline

## One-time project setup

1. Open `Visual-Imagination-Perception-EEG-Decoding` as the PyCharm project.
2. Go to **Settings > Project > Python Interpreter**.
3. Add a new Virtualenv interpreter using Python 3.11 at `.venv`.
4. Open the PyCharm terminal.
5. Run `python -m pip install --upgrade pip`.
6. Run `pip install -r requirements.txt`.
7. Run `pip install -e . --no-deps`.
8. Use the repository root as the working directory for every run configuration.

## Run configurations

Create Python configurations with these exact values:

| Name | Script path | Parameters |
|---|---|---|
| 00 - Verify installation | `scripts/verify_installation.py` | `--config configs/paper.yaml --output results/verification.json --verbose` |
| 01 - Download data | `scripts/download_data.py` | `--config configs/paper.yaml --verbose` |
| 02 - Prepare 6,000 topomaps | `scripts/prepare_topomaps.py` | `--config configs/paper.yaml --verbose` |
| 03 - Random forest | `scripts/run_random_forest.py` | `--config configs/paper.yaml --verbose` |
| 04 - Difference statistics | `scripts/run_statistics.py` | `--config configs/paper.yaml --verbose` |
| 05 - Paper 15-fold CNN | `scripts/run_cnn_loso.py` | `--config configs/paper.yaml --grouping subject_session --verbose` |
| 06 - True participant LOSO | `scripts/run_cnn_loso.py` | `--config configs/paper.yaml --grouping subject --verbose` |
| 07 - Publication analysis | `scripts/reproduce_paper_analysis.py` | `--config configs/paper.yaml --verbose` |
| 08 - Figures only | `scripts/make_figures.py` | `--config configs/paper.yaml --verbose` |
| 99 - Complete paper pipeline | `scripts/run_all.py` | `--config configs/paper.yaml --verbose` |

No environment variables are required.

To run only the paper's `18_1` held-out fold, append `--fold 18_1` to configuration 05. All of its artifacts are isolated under `selected_folds/18_1`; the full-run folds and summary are not created or overwritten. To rebuild an existing stage for the unchanged configuration, append `--force`.
