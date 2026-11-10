# Code walkthrough (≈ 700 lines in total)

```
tada/data.py        paths, the 19 ASVspoof generators, clip list, train/test splits, feature loading
tada/audio.py       load audio + 8 channel degradations (sampling, quantisation, noise, filters, MP3)
tada/features.py    MFCC, LFCC, LPCC (+ Levinson-Durbin) and the frozen w2v-BERT 2.0 model
tada/models.py      kNN, GMM, OOD score, EER
scripts/1_download_data.py      get a speaker-balanced subset of ASVspoof 2019 LA
scripts/2_extract_features.py   one feature vector per clip (and per degraded copy)
scripts/3_experiments.py        attribution, unseen-generator detection, robustness -> results/results.json
scripts/4_figures.py            all figures in results/figures/
demo.py                         live demo on one clip
tests/test_core.py              6 checks against textbook results
```

## 1. Data: `tada/data.py`, `scripts/1_download_data.py`

* `SYSTEMS` lists the 19 ASVspoof 2019 LA generators with their vocoder. A01–A06 appear
  in train/dev, A07–A19 only in eval. **A16 = A04 and A19 = A06** (same algorithms).
* The download script never fetches the full 7.6 GB. The parquet files are made of
  100-row "row groups". We read only the metadata columns, **plan** a speaker-balanced
  sample (`plan()`: a row is accepted only while its class needs clips and its speaker
  has not used up its share), and then download only the row groups we need.
* `random_split` is an 80:20 split stratified by class. `speaker_split` holds out 20 % of
  the **speakers**, so the test speakers are never in the support set.

## 2. Features: `tada/features.py`

All classical features follow the course recipe:
pre-emphasis (0.97) → 25 ms frames / 10 ms hop → window → |DFT|² → filterbank or LPC →
coefficients per frame → Δ, ΔΔ → **mean and std over frames** (one vector per clip).

| function | idea | size |
|---|---|---|
| `mfcc` | 40 **mel**-spaced filters → log → DCT → 13 coefficients | 78 |
| `lfcc_frames` | the same with **linearly** spaced filters (`linear_filterbank`) | 120 |
| `lfcc_frames(y, 10, 4000, 8000)` | LFCC using only the 4–8 kHz band | 60 |
| `lpcc` | autocorrelation → `levinson` (solves R a = −r) → `lpc_to_cepstrum` + log residual energy | 34 |

`W2VBert` wraps the frozen pre-trained model. `prepare` computes the 80-band log-mel
input on the CPU. `embed` runs the 24 Conformer blocks on the GPU and returns the time
average of all 25 hidden states. We use layer 4, as TADA does.

## 3. Models: `tada/models.py`

* `knn_predict`: majority vote of the 21 nearest support vectors. **Nothing is trained**;
  the "model" is the stored support set. Classical features are z-scored first
  (`zscore`), because their dimensions have very different scales.
* `GMMClassifier`: one 32-component diagonal Gaussian mixture per generator, fitted on
  frame-level LFCCs by EM. The prediction is the generator with the highest average
  log-likelihood. This is the classical (course) statistical model.
* `ood_score`: mean distance to the 21 nearest *known* clips. Large means unknown generator.
* `eer`: the threshold where false alarms equal misses.

## 4. Experiments: `scripts/3_experiments.py`

* `attribution()`: 13 generators (A07–A19), 400 clips each, 5 random and 5
  speaker-disjoint splits. Classical vs w2v-BERT, the GMM baseline, and a sweep over all
  25 layers.
* `unseen_generators()`: the support set is A01–A06 (train partition). The known test is
  A01–A06 from the dev partition (new speakers); the unknown set is A07–A19. Reports
  AUROC per system and "which known generator does each unseen one land on?".
* `robustness()`: clean support set; test clips degraded by each channel from `tada/audio.py`.

## 5. Demo: `demo.py`

One clip → w2v-BERT layer 4 → 21 nearest neighbours among the 5,199 other eval clips →
vote. If the clip is farther from the bank than 99 % of the bank's own clips, it prints
"UNKNOWN generator". Writes `demo_output.png` (waveform, spectrogram, votes).
