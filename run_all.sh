#!/usr/bin/env bash
# Full pipeline. Run from the code/ folder.
set -e
python scripts/1_download_data.py      # ASVspoof 2019 LA subset (~0.6 GB)
python scripts/2_extract_features.py   # MFCC/LFCC/LPCC + w2v-BERT (GPU recommended)
python scripts/3_experiments.py        # -> results/results.json + tables
python scripts/4_figures.py            # -> results/figures
