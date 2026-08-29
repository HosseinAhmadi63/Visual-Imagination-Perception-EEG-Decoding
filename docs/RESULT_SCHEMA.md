# Result schema

## Topomap manifest

`data/topomaps/manifest.csv` has one row per PNG:

| Column | Meaning |
|---|---|
| `path` | Repository-relative PNG path |
| `subject` | Participant numeric ID |
| `session` | Session numeric ID |
| `subject_session` | Table II recording key |
| `task` | `perception` or `imagination` |
| `label` | 0 for Perception, 1 for Imagination |
| `frame_index` | Zero-based frame index, 0-199 |
| `time_seconds` | Time from the selected state marker |
| `source_fif` | Repository-relative curator FIF path |
| `source_event` | Normalized event description |
| `source_onset_seconds` | Event onset relative to the continuous recording |
| `stimulus` | Category/complexity suffix used to pair the two markers |

## Random-forest predictions

`test_predictions.csv` retains every manifest column and adds:

- `manifest_index`
- `probability_imagination`
- `prediction`

`summary.json` contains holdout metrics, train/test sizes, feature dimensions, permutation result, PCA explained variance, silhouette score, and cluster count.

## CNN fold metrics

`fold_metrics.csv` contains one row per held-out group:

- `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- `samples`, `perception_samples`, `imagination_samples`
- `test_loss`
- `best_epoch`, `best_validation_loss`, `epochs_completed`
- `training_images`, `validation_images`, `test_images`
- `seed`, `device`, `fold`, `grouping`, `test_subject_sessions`

Each fold's `predictions.csv` retains the manifest provenance and adds `fold`, `grouping`, `probability_imagination`, `prediction`, and `correct`.

`aggregate_predictions.csv` concatenates held-out rows. `aggregate_roc.csv` contains false-positive rate, true-positive rate, and threshold. `summary.json` contains the unweighted mean and sample standard deviation of fold metrics plus aggregate image-level metrics.

When `--fold` selects a subset, the same schema is written under `selected_folds/<ordered-selection>/`, including separate fold checkpoints and predictions. A selected-fold run never writes the canonical complete-grouping `summary.json`.

## Cluster statistics

`channel_cluster_statistics.csv` has one row per task and EEG channel:

- `task`
- `channel`
- `t_statistic`
- `minimum_cluster_p`
- `significant`

`difference_statistics.npz` preserves the 15 recording-level difference arrays, mean differences, t statistics, minimum cluster p-values, channel names, montage positions, and sampling rate.

## Publication outputs

`paper_consistency.json` combines source-table integrity checks, all completed analysis summaries, and the figure manifest. Generated CSV files are prefixed with `generated_` to distinguish them from immutable article transcriptions.
