# Data

The pipeline uses the curator-preprocessed release of:

- OpenNeuro accession: `ds004306`
- Snapshot: `1.0.0`
- Dataset DOI: [10.18112/openneuro.ds004306.v1.0.0](https://doi.org/10.18112/openneuro.ds004306.v1.0.0)
- Web page: [https://openneuro.org/datasets/ds004306/versions/1.0.0](https://openneuro.org/datasets/ds004306/versions/1.0.0)
- License: CC0

The downloader requests only `derivatives/preprocessed`. That revision contains 15 main FIF files and 11 MNE split-file continuations, 26 objects totaling 33,249,937,247 bytes, or 30.97 GiB. MNE opens each main file and automatically discovers its colocated `-1.fif` continuation.

After a complete run:

```text
data/openneuro/derivatives/preprocessed/   Versioned curator FIF files
data/epochs/                               One compact paired-epoch archive per subject-session
data/topomaps/                             6,000 generated 150 x 150 RGB PNG files
data/topomaps/manifest.csv                 Complete source-to-image provenance
```

Download and verify all required FIF files:

```bash
python scripts/download_data.py --config configs/paper.yaml --verbose
```

Verify an existing download without network transfer:

```bash
python scripts/download_data.py --config configs/paper.yaml --validate-only --verbose
```

The FIF files retain 124 EEG channels at 1024 Hz, embedded head-coordinate electrode locations, and the dataset curator's filtering, bad-channel correction, common-average referencing, and FastICA artifact-removal result. The paper starts from these preprocessed files; this pipeline does the same and does not silently repeat preprocessing.

Raw EEG, epoch archives, generated topomaps, and trained weights are excluded from Git.
