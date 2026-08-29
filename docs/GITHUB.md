# Publish to GitHub

From the repository root, authenticate GitHub CLI once and publish the complete project:

```bash
gh auth login
git init
git add .
git commit -m "Add complete EEG topomap reproduction pipeline"
git branch -M main
gh repo create HosseinAhmadi63/Visual-Imagination-Perception-EEG-Decoding --public --source=. --remote=origin --push --description "Complete reproduction pipeline for decoding visual imagination and perception from EEG topomap sequences."
```

The resulting repository URL is:

[https://github.com/HosseinAhmadi63/Visual-Imagination-Perception-EEG-Decoding](https://github.com/HosseinAhmadi63/Visual-Imagination-Perception-EEG-Decoding)

Raw FIF files, generated PNGs, checkpoints, and run outputs remain excluded by `.gitignore`.
