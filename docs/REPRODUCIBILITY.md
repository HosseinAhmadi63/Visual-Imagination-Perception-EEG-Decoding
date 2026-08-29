# Reproducibility contract

## Immutable inputs

- Paper DOI: `10.1109/EMBC58623.2025.11251641`
- Dataset DOI: `10.18112/openneuro.ds004306.v1.0.0`
- OpenNeuro snapshot: `1.0.0`
- Object inventory: `data/openneuro_preprocessed_manifest.csv`
- Paper values: `results/publication/source`
- Experiment definition: `configs/paper.yaml`

The object inventory records all 26 relative paths, byte counts, SHA-256 values, and version IDs. The downloader requests the named snapshot and independently verifies remote hashes and sizes. Local validation checks all 26 paths and byte counts before any EEG processing.

## Determinism

The project seed is 42. Python, NumPy, scikit-learn, PyTorch CPU/CUDA seeds, and deterministic cuDNN settings are configured. Every CNN fold adds its zero-based fold index to the project seed, so folds have distinct but stable initialization and data order.

The exact dependency environment is in `requirements.txt`. `pyproject.toml` provides compatible ranges for normal installation.

GPU kernels and third-party library builds can still produce small platform-level numerical differences. The verification command records dependency versions and the configuration hash. Run directories include that configuration hash, and every CNN fold records its seed, device, best epoch, predictions, and history so differences can be traced.

## Resumption

The 12-character run key is the SHA-256 prefix of the complete YAML configuration. Existing fold metrics, predictions, and checkpoints under that key are reused. `--force` recomputes the selected stage without changing the run key.

OpenNeuro downloads resume at file level. Topomap generation validates existing PNGs and atomically replaces missing or damaged images. CNN folds resume independently, and selected-fold test runs are isolated from complete-grouping artifacts.

## Verification commands

```bash
pytest
ruff check .
vip-eeg verify-paper --config configs/paper.yaml
vip-eeg verify --config configs/paper.yaml --output results/verification.json
python scripts/download_data.py --config configs/paper.yaml --validate-only --verbose
```

The first three commands require no EEG data. The final command validates the complete local OpenNeuro subset.
