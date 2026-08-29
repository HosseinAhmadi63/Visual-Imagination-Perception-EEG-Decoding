# Implementation details

## Source data

The paper starts from the dataset curator's preprocessed continuous FIF files. Snapshot `1.0.0` contains 124 EEG channels at 1024 Hz, custom head-coordinate locations embedded in each FIF, a 1 Hz high-pass metadata value, 512 Hz low-pass metadata value, and common-average reference state.

The [public preprocessing source pinned at commit](https://github.com/hWils/Semantics-EEG-Perception-and-Imagination/tree/60dfbfec90adf0d5a4646f64ee6af203ebf5f2b7) loads the 1024 Hz EEGLAB recording, marks four acquisition channels as EOG and retains the 124 EEG channels, installs a custom head-frame montage, detects automatic bad channels with PyPREP without RANSAC, applies average reference, interpolates bad channels, applies average reference again, applies zero-double FIR notch filters at 50, 100, and 150 Hz, high-pass filters at 1 Hz, applies average reference a third time, fits 50-component MNE ICA with seed 97 and maximum 800 iterations, detects ocular components from Fp1/Fp2-derived epochs at threshold 1.96, applies the ICA exclusion, and saves the continuous FIF. The preprocessed FIF already embeds the custom montage, so no unavailable external coordinate file is required downstream. The OpenNeuro object inventory was audited against Git snapshot.

The upstream notebook accepts a manual bad-channel list but does not add it to `raw.info["bads"]`; only the automatic PyPREP list reaches interpolation. This repository consumes the published preprocessed result and does not reinterpret that upstream behavior.

## Deterministic trial selection

The paper specifies one pictorial trial per subject but does not publish the selected repetitions. Exact private selection recovery is therefore impossible. The repository applies one immutable rule:

1. Convert FIF annotations to sample-ordered events.
2. Remove a leading `0, ` and trailing `###my_stream_name` from every description.
3. Find the earliest `Perception_image_*` event whose immediately following annotation is `Imagination_image_*` with the identical category/complexity suffix.
4. Use each event's independent marker sample as its epoch start.
5. Verify the resulting samples against `results/publication/source/deterministic_trial_selection.csv`.

String matching is required because numeric event values vary between recordings. The category suffix is flower, guitar, or penguin. Complexity `s`, `m`, and `c` means simple, intermediate, and complex/naturalistic.

Epoch archives retain the MNE endpoint sample at 3.0 or 4.0 seconds. The 200 model frames use the left-closed grids `k * 0.015` and `k * 0.020` for `k=0..199`, so the final model times are 2.985 and 3.980 seconds.

## Topomap images

Every model PNG contains only the scalp interpolation, head outline, contours, and sensor markers. It contains no title, task name, time, filename, axes, or colorbar. This prevents rendered label leakage. Task and provenance are stored only in `manifest.csv`.

The paper does not state a color-normalization scope. The implementation freezes a symmetric range independently for every frame, matching the normal single-map MNE diverging-topomap behavior. The research figures use a shared scale across the panels they compare. Values remain in volts for model PNG generation; publication figures convert to microvolts before labeling their colorbars.

## CNN translation

Figure 3 determines the architecture. Five floor-mode max-pools transform 150 to 75, 37, 18, 9, and 4 pixels. Same convolutions therefore use padding 1. The block order is Conv2d, ReLU, BatchNorm2d, squeeze-and-excitation, element-wise dropout, MaxPool2d. The head order follows Figure 3: dense, ReLU, BatchNorm1d, dropout, output.

The PyTorch model returns one logit and uses `BCEWithLogitsLoss`; sigmoid is applied once for metrics. Convolutions omit bias because every convolution is followed by batch normalization. The paper does not report the SE reduction, weight-decay coefficient, or validation split. They are frozen as 16, `1e-4`, and a stratified 10% frame split of the 14-recording development set. This validation choice can place adjacent frames from one selected trial in both training and validation; it is a model-selection subset, not independent subject-level validation. Adam otherwise uses PyTorch defaults. Validation loss controls checkpointing, early stopping, and learning-rate reduction.

The paper repeatedly defines Perception as 0 and Imagination as 1, including the sigmoid's semantic meaning. Figure 3 reverses the printed class names. Code follows the repeated method/results definition.

## Evaluation groups

Table II's columns are 15 subject-session recordings from 12 people. `subject_session` reproduces those columns. In each fold, 400 images are tested and 5,600 images are available for development before the fixed validation split. The paper's written `14 x 200 = 2800` count is the per-class count.

Participants 12, 14, and 15 each have two sessions. The paper grouping allows their other session into training. The secondary `subject` grouping holds all sessions of one person out together and provides true participant-level LOSO.

All aggregated ROC points come from concatenating exactly one held-out prediction per image across the completed folds.

## Preliminary analyses

The RF analysis intentionally uses the paper's frame-level stratified 80/20 split. Its 1,200-image test support is balanced at 600 per class. RF hyperparameters were not stated and are frozen to 100 scikit-learn default-style trees with square-root feature sampling and seed 42.

Figure 6 calls the observed result cross-validated while the RF prose describes one holdout. The implementation retains both: `holdout.accuracy` reports the 80/20 result, and `permutation.observed_cross_validated_accuracy` reports the five-fold score used by 100 label permutations. The p-value uses scikit-learn's finite-permutation estimator.

PCA receives the flattened RGB values on their saved 0-255 scale, matching Figure 7's coordinate magnitude, uses randomized two-component SVD with seed 42, and is followed by two-cluster Ward agglomeration. A constant rescaling would not alter explained-variance ratios, clusters, or silhouette score, but retaining the saved scale also preserves comparable plot axes. The silhouette score is computed in the two-dimensional PCA space.

## Difference statistics

The paper caption names a cluster-based permutation test but supplies no design. The repository freezes a two-sided one-sample channel-cluster test across the 15 subject-session differences. Electrode adjacency is computed from the embedded montage, the cluster-forming threshold is MNE's automatic t threshold, 1,024 permutations are used, and cluster alpha is 0.05. This completes the stated analysis without inventing a significance claim.
