# Who made this voice? Training-free attribution of audio deepfakes

AI553 Speech & Audio Processing, course project. Saaransh Garg (2024AIB1011) and [Partner name].

We study **TADA** (Stan et al., Interspeech 2025): a frozen w2v-BERT 2.0 model (layer 4,
averaged over time) plus k-nearest neighbours, with **no training**. We compare it with
course features (MFCC, LFCC, LPCC, GMM) on 9,800 real clips from 19 ASVspoof 2019 LA
generators.

| Result | Value |
|---|---|
| Attribution of 13 generators, w2v-BERT L4 + kNN | **97.6 %** (97.0 % on unseen speakers) |
| MFCC / LFCC / LFCC 4–8 kHz only | 72.2 % / 76.4 % / 84.3 % |
| GMM on LFCC frames (trained) | 96.6 % |
| Detecting unseen generators (AUROC) | 96.6 % (MFCC 77.9 %) |
| A16 → A04 / A19 → A06 (same algorithms) | 100 % / 98 % |
| After 8 kHz resampling / μ-law / MP3 / 20 dB noise | 92 / 95 / 91 / 52 % |

## Contents
- `report.pdf`: the report (LaTeX source in `report_source/`)
- `presentation.pptx`: 11 slides for the 10-minute talk, with speaker notes
- `AI_DISCLOSURE.md`: disclosure of AI-tool use
- `code/`: `tada/` (4 small modules), `scripts/1-4`, `demo.py`, `tests/`
- `results/`: `results.json` (every number), figures, tables
- `demo_clips/`: a human clip and a 2025 TTS clip for the live demo

## Run
```
cd code
pip install -r requirements.txt     # torch 2.6 first (CUDA or CPU build)
bash run_all.sh                     # download -> features (~15 min on GPU) -> experiments -> figures
python demo.py                      # needs the extracted features (data/ and features/)
python demo.py --file ../demo_clips/human.flac
python -m pytest -q tests
```
