# Results

`publication/source` contains immutable values transcribed from the article. These files are versioned and are never overwritten by a run.

`runs/<config-hash>` contains resumable stage outputs, predictions, histories, and checkpoints for one exact configuration.

`publication/generated` contains paper comparisons, generated tables, and paper-style figures assembled from the active configuration's completed run.

Generated files are excluded from Git except for the two directory keep files. Commit selected results only after verifying their provenance and size.
