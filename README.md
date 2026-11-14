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
- `demo_clips/`: human and modern TTS clips, plus one labeled reference example per class (A07–A19 and bonafide)
- `dashboard.py`: local browser demo with an interactive 2D embedding map

## Run
Run from the directory containing `demo.py` (this checkout's root; `code/` in the submission).
```bash
source .venv/bin/activate           # existing local environment
pip install -r requirements.txt    # torch 2.6 first (CUDA or CPU build)
bash run_all.sh                    # download -> features (~15 min on GPU) -> experiments -> figures
python demo.py                     # needs extracted data/ and features/
python demo.py --file demo_clips/human.flac
python demo.py --file demo_clips/A07.flac
python -m pytest -q tests
```

## Browser dashboard
```bash
source .venv/bin/activate
streamlit run dashboard.py --server.address localhost --browser.gatherUsageStats false
```
Open **http://localhost:8501**. Select a bundled demo clip or upload WAV, FLAC, OGG, or MP3, then press **Locate audio**. Recordings must be 0.2–30 seconds and at most 20 MB. Use WAV or FLAC if another format cannot be decoded.

- Gray circles: labeled reference bank; blue circles/links: 21 neighbors; orange diamond: selected audio. Hover, zoom, or pan; tables provide the same reference details.
- PCA is fitted once on the reference bank. New embeddings use that same transform. **Neighbors and voting use the original 1,024-dimensional embeddings, not 2D distances.**
- Curated reference clips use stored embeddings for fast demonstrations. Byte-identical copies (including renamed uploads) are recognized and their UID excluded from voting. Re-encoded copies are not recognized.
- Custom recordings use frozen w2v-BERT locally. First model load can take time and may download weights if not already cached. Audio is not sent to an external service.
- The dashboard reports the closest known class and votes, **not confidence or verified origin**. It does not apply `demo.py`'s unknown-generator distance heuristic.

`demo_clips/reference_clips.csv` records each curated clip's UID, label, speaker, hash and source. Examples were sampled deterministically, not selected by prediction correctness. These are reference-bank clips, not independently held-out tests; the encoder itself is not trained here.
