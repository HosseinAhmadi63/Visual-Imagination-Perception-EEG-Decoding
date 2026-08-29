# Configuration

[`paper.yaml`](paper.yaml) is the complete frozen definition of the paper reproduction. It contains the immutable dataset snapshot, the 15 Table II subject-session keys, event matching, epoch timing, frame sampling, topomap rendering, CNN architecture, training callbacks, RF analysis, clustering, statistics, labels, paths, and DOI metadata.

Changing any value changes the deterministic 12-character result key, so outputs from different protocols never share a run directory.
