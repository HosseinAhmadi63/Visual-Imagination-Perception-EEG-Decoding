# Paper-to-code map

| Paper item | Implementation | Output |
|---|---|---|
| Section III-A, dataset selection | `vip_eeg.data.openneuro` | Validated OpenNeuro snapshot under `data/openneuro` |
| Section III-A, one pictorial trial | `vip_eeg.data.events.select_first_pictorial_pair` | Frozen selection checked against `deterministic_trial_selection.csv` |
| Section III-A, 3 s/4 s epochs | `vip_eeg.data.prepare.extract_epoch` | `data/epochs/<subject_session>.npz` |
| Section III-B, 200 frames/state | `vip_eeg.data.prepare.frame_offsets` | Exactly 400 manifest rows per recording |
| Section III-B, MNE topomaps | `vip_eeg.data.topomaps.render_topomap` | 6,000 clean RGB PNGs and `manifest.csv` |
| Figure 2 | `vip_eeg.plotting.paper.example_topomaps` | `figure_2_selected_topomaps.png` |
| Figure 3 | `vip_eeg.models.cnn.TopomapSECNN` | Five exact intermediate shapes verified by tests |
| Section III-C, training | `vip_eeg.evaluation.cnn_loso.train_fold` | History, best checkpoint, predictions, metrics per fold |
| Section III-C/IV-B, reported folds | `run_cnn_loso(..., grouping="subject_session")` | 15 Table II-aligned folds |
| Strict LOSO | `run_cnn_loso(..., grouping="subject")` | 12 participant-disjoint folds |
| Section IV-A, RF | `vip_eeg.evaluation.random_forest.run_random_forest` | Holdout report, confusion matrix, ROC, predictions |
| Figure 4 | `vip_eeg.plotting.paper.difference_topomaps` | Subject-session `18_1` difference maps |
| Figure 4 caption, cluster test | `vip_eeg.evaluation.statistics.run_cluster_statistics` | Channel statistics, clusters, corrected p-values |
| Figure 5 | `vip_eeg.plotting.paper.random_forest_roc` | RF ROC figure |
| Figure 6 | `permutation_test_score` and `permutation_histogram` | 100 scores, p-value, histogram |
| Figure 7 | PCA plus `AgglomerativeClustering` | Coordinates, cluster IDs, silhouette score, scatter |
| Figure 8 | Concatenated out-of-fold predictions | Aggregated ROC CSV and figure |
| Table I | `classification_report` | Generated RF class report and reference comparison |
| Table II | `fold_metrics.csv` and source table | Fold accuracy comparison and computed averages |

The execution order is encoded in `vip_eeg.cli.run_complete_pipeline` and exposed by `scripts/run_all.py`.
