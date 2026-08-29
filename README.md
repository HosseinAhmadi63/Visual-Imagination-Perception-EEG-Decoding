# Visual Imagination and Perception EEG Decoding

This repository contains the complete Python/PyCharm reproduction pipeline for:

> **Decoding Visual Imagination and Perception from EEG via Topomap Sequences**  
> Hossein Ahmadi and Luca Mesin  
> *2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)*  
> [https://doi.org/10.1109/EMBC58623.2025.11251641](https://doi.org/10.1109/EMBC58623.2025.11251641)

The pipeline downloads the versioned OpenNeuro EEG release, selects one matched pictorial Perception/Imagination pair from every reported recording, creates the paper's 6,000 MNE topomap PNGs, runs the random-forest and permutation baselines, performs PCA and agglomerative clustering, evaluates the five-block CNN with squeeze-and-excitation under the reported 15 recording folds, runs the stated cluster-based difference-map test, and generates equivalents of Figures 1, 2, and 4-8 plus Tables I and II.

If you use this repository, its code, or its results, cite the article above. Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Frozen implementation protocol

The table separates values stated by the article from deterministic completion choices required where the article is silent. All completion choices are documented and versioned; they are not presented as recovered author settings.

| Stage | Implemented protocol |
|---|---|
| Dataset | OpenNeuro `ds004306`, immutable snapshot `1.0.0`, curator-provided preprocessed FIF files |
| Paper cohort | `3_3`, `8_3`, `10_1`, `11_1`, `12_1`, `12_2`, `13_1`, `14_1`, `14_2`, `15_1`, `15_2`, `16_1`, `17_1`, `18_1`, `19_1` |
| Pictorial trial | Repository choice: earliest chronologically matched `Perception_image_*` event and following `Imagination_image_*` event with the same stimulus suffix |
| Epochs | Perception 3.0 s, label 0; Imagination 4.0 s, label 1 |
| Frames | 200 per state and recording; 0.015 s Perception interval and 0.020 s Imagination interval |
| Topomaps | Article: MNE scalp interpolation and 150 x 150 RGB PNG. Repository choices: `RdBu_r`, 6 contours, sensor markers, symmetric per-frame color range, and label-free rendering |
| Image total | 15 recordings x 2 states x 200 frames = 6,000 images |
| RF branch | Article: 80/20 frame split, 100 permutations, PCA to two dimensions, and two-cluster agglomeration. Repository choices: stratification, 100 trees, five-fold permutation CV, and Ward linkage |
| CNN blocks | Filters 32, 64, 128, 256, 512; 3 x 3 same convolution with ReLU, BN, SE, dropout 0.1, 2 x 2 max pooling |
| CNN head | Flatten 4 x 4 x 512, dense 256 with ReLU, BN, dropout 0.3, one logit with sigmoid probability at evaluation |
| Optimization | Article: Adam, LR `1e-4`, L2 regularization, batch 32, maximum 50 epochs. Repository choice: weight decay `1e-4` |
| Callbacks | Validation-loss checkpoint, early stopping patience 5, LR plateau patience 3, factor 0.5, minimum LR `1e-6` |
| Paper evaluation | 15 leave-one-subject-session-out folds, 5,600 development images and 400 test images before the validation split |
| Secondary evaluation | True 12-participant LOSO, enabled with one exact command below |
| Difference test | Article: cluster-based permutation test at 0.400/0.415 s and 0.400/0.420 s. Repository design: 1,024 two-sided channel-cluster permutations across the 15 recordings |

Every fixed value is in [`configs/paper.yaml`](configs/paper.yaml). The direct article-to-code map is in [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md).

## Repository structure

```text
configs/paper.yaml                    Frozen experiment definition
scripts/download_data.py              OpenNeuro snapshot download and validation
scripts/prepare_topomaps.py            Trial selection, epoch archives, 6,000 PNGs
scripts/run_random_forest.py           RF, holdout ROC, permutations, PCA, clustering
scripts/run_statistics.py              Difference maps and channel-cluster tests
scripts/run_cnn_loso.py                Paper folds or true participant LOSO
scripts/reproduce_paper_analysis.py    Tables, comparisons, and paper-style figures
scripts/run_all.py                     Complete resumable pipeline
src/vip_eeg/                           Installable implementation package
results/publication/source/            Values transcribed from Tables I-II and figures
results/publication/generated/         Generated publication tables and figures
results/runs/<config-hash>/             Predictions, histories, checkpoints, and metrics
tests/                                  Scientific and implementation invariants
```

## Installation

Python 3.11 is the repository's frozen reference interpreter; the paper does not report a Python version.

```bash
git clone https://github.com/HosseinAhmadi63/Visual-Imagination-Perception-EEG-Decoding.git
cd Visual-Imagination-Perception-EEG-Decoding
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

For a convenient editable development environment that resolves compatible dependency ranges instead:

```bash
pip install -e ".[dev]"
```

Verify the configuration, source tables, CNN dimensions, forward pass, and backward pass without downloading EEG data:

```bash
pytest
vip-eeg verify-paper --config configs/paper.yaml
vip-eeg verify --config configs/paper.yaml --output results/verification.json
```

## PyCharm

1. Open the repository root in PyCharm.
2. Select Python 3.11 and create the project interpreter at `.venv`.
3. Open the PyCharm terminal and run `pip install -r requirements.txt`, followed by `pip install -e . --no-deps`.
4. Create a Python run configuration with script `scripts/run_all.py`.
5. Set the working directory to the repository root.
6. Set parameters to `--config configs/paper.yaml --verbose`.
7. Run the configuration.

All individual stage configurations are listed in [`docs/PYCHARM.md`](docs/PYCHARM.md).

## Complete reproduction

The preprocessed FIF subset is 30.97 GiB. Keep at least 45 GiB free for the download, topomaps, checkpoints, and results. The download resumes after interruption and verifies both size and hash.

Run every paper stage in order:

```bash
python scripts/run_all.py --config configs/paper.yaml --verbose
```

When `data/openneuro` already contains snapshot `1.0.0`:

```bash
python scripts/run_all.py --config configs/paper.yaml --skip-download --verbose
```

Run the reported 15 recording folds and the stricter 12-participant LOSO analysis together:

```bash
python scripts/run_all.py --config configs/paper.yaml --strict-participant-loso --verbose
```

Every stage is resumable. Completed outputs are reused. `--force` intentionally replaces outputs for the same frozen configuration.

## Explicit stage sequence

```bash
python scripts/download_data.py --config configs/paper.yaml --verbose
python scripts/prepare_topomaps.py --config configs/paper.yaml --verbose
python scripts/run_random_forest.py --config configs/paper.yaml --verbose
python scripts/run_statistics.py --config configs/paper.yaml --verbose
python scripts/run_cnn_loso.py --config configs/paper.yaml --grouping subject_session --verbose
python scripts/reproduce_paper_analysis.py --config configs/paper.yaml --verbose
```

Run one recording through topomap preparation:

```bash
python scripts/prepare_topomaps.py --config configs/paper.yaml --recording 18_1 --verbose
```

Run one reported CNN fold. Its aggregate files, predictions, history, and checkpoint are isolated under `selected_folds/18_1` and cannot be mistaken for or overwrite a complete run:

```bash
python scripts/run_cnn_loso.py --config configs/paper.yaml --grouping subject_session --fold 18_1 --verbose
```

Run true participant-level LOSO:

```bash
python scripts/run_cnn_loso.py --config configs/paper.yaml --grouping subject --verbose
```

## Outputs

The configuration receives a deterministic 12-character SHA-256 run key. Principal outputs are:

```text
data/epochs/<subject_session>.npz
data/topomaps/manifest.csv
data/topomaps/<subject_session>/<task>/*.png
results/runs/<run-key>/random_forest/summary.json
results/runs/<run-key>/random_forest/test_predictions.csv
results/runs/<run-key>/random_forest/permutation_scores.csv
results/runs/<run-key>/random_forest/pca_clusters.csv
results/runs/<run-key>/statistics/channel_cluster_statistics.csv
results/runs/<run-key>/cnn_subject_session/fold_metrics.csv
results/runs/<run-key>/cnn_subject_session/folds/*/history.csv
results/runs/<run-key>/cnn_subject_session/folds/*/predictions.csv
results/runs/<run-key>/cnn_subject_session/folds/*/best_model.pt
results/runs/<run-key>/cnn_subject_session/aggregate_roc.csv
results/publication/generated/paper_consistency.json
results/publication/generated/figures/*.png
```

Exact columns and semantics are documented in [`docs/RESULT_SCHEMA.md`](docs/RESULT_SCHEMA.md).

## Reproducibility scope

The article does not identify which one of the many pictorial repetitions was selected. This repository freezes the earliest matched pair as a deterministic, auditable rule and records both source event names and onsets in every manifest row. It does not claim that this unrecoverable choice is the authors' original private selection.

Table II contains 15 subject-session recordings but only 12 unique participants. The default `subject_session` grouping matches the table's 15 recording folds and compares newly generated accuracies against the 15 transcribed article values. The supported `subject` grouping keeps both sessions of participants 12, 14, and 15 together and therefore removes cross-session participant leakage.

The repeated method text defines Perception as 0 and Imagination as 1; Figure 3 reverses those two labels. The implementation follows the method and results text. The paper's written per-fold training count omits the second class: 14 x 200 is 2,800 images per class and 5,600 images in total.

The title uses “sequences,” but Figure 3 and the method classify one topomap frame at a time. The implementation therefore contains the stated frame CNN and does not add an unreported recurrent or temporal-sequence model. The paper leaves the topomap color-normalization scope, SE reduction, L2 coefficient, validation fraction, RF hyperparameters, permutation CV folds, and cluster-test design unstated. The repository freezes these as per-frame symmetric scaling, 16, `1e-4`, 10%, 100 default-style RF trees, five stratified folds, and a two-sided one-sample channel-cluster test with 1,024 permutations. These completion choices are versioned and never inferred at run time.

The RF split is intentionally frame-level because that is the paper's stated 80/20 experiment. Adjacent frames from the same epoch can occur on both sides of that split, so RF accuracy and its label-permutation p-value are preliminary frame-level evidence, not subject-generalization estimates. The frozen CNN validation subset is also frame-stratified within each development fold and can share adjacent frames with training; it controls model selection but is not an independent validation cohort. The CNN held-out recording-fold and participant-fold test results are the generalization evaluations.

## Data and licenses

No EEG recording is committed. The pipeline downloads OpenNeuro dataset `ds004306`, snapshot `1.0.0`, DOI [10.18112/openneuro.ds004306.v1.0.0](https://doi.org/10.18112/openneuro.ds004306.v1.0.0). The dataset is CC0. Dataset details and exact local paths are in [`data/README.md`](data/README.md).

## Citation

```bibtex
@inproceedings{ahmadi2025topomapsequences,
  author    = {Ahmadi, Hossein and Mesin, Luca},
  title     = {Decoding Visual Imagination and Perception from EEG via Topomap Sequences},
  booktitle = {2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  year      = {2025},
  pages     = {1--7},
  doi       = {10.1109/EMBC58623.2025.11251641}
}
```

The source dataset should also be cited:

```bibtex
@article{wilson2023semanticdataset,
  author  = {Wilson, Holly and Golbabaee, Mohammad and Proulx, Michael J. and Charles, Stephen and O'Neill, Eamonn},
  title   = {EEG-based BCI Dataset of Semantic Concepts for Imagination and Perception Tasks},
  journal = {Scientific Data},
  year    = {2023},
  volume  = {10},
  pages   = {386},
  doi     = {10.1038/s41597-023-02287-9}
}
```

## License

The source code is released under the [MIT License](LICENSE). The OpenNeuro dataset's CC0 terms and the IEEE article's terms remain separate and controlling.
