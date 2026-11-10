# Viva preparation: likely questions and short answers

All numbers come from `results/results.json`.

## The problem
**Detection vs attribution?** Detection asks "real or fake?". Attribution asks "*which*
generator made it?", and in practice it is open-set: we must also be able to say
"unknown generator".

**Why can generators be told apart?** Each generator is a fixed pipeline (acoustic model +
vocoder), and every stage leaves consistent artefacts: band limits, buzzy source-filter
excitation, Griffin-Lim phase errors, noise floors. Those artefacts act as a fingerprint.

**What does "training-free" mean? w2v-BERT was trained.** Meta pre-trained it once on
unlabelled speech. *We* never train or fine-tune anything. kNN just stores the support
vectors, so adding a generator means adding its vectors.

## Signal processing (course)
**MFCC pipeline?** Pre-emphasis (y[n] = x[n] − 0.97x[n−1]) → 25 ms Hann frames, 10 ms hop
→ |DFT|² → 40 mel filters → log → DCT → 13 coefficients → Δ, ΔΔ → mean and std over
frames = 78 numbers.

**Why does LFCC beat MFCC, and 4–8 kHz LFCC (84.3 %) beat full-band LFCC (76.4 %)?** The mel
scale copies human hearing and squeezes the high frequencies into a few wide filters.
Vocoder artefacts are most visible at 4–8 kHz, where speech energy is low. Linear
filters keep that detail, and dropping the low band removes speaker and phonetic
content that dilutes it.

**LPC / Levinson–Durbin?** Model each frame as an all-pole filter 1/A(z) driven by a
residual. Minimising the prediction error gives the Toeplitz equations R a = −r, which
Levinson–Durbin solves in O(p²). We convert to cepstra (LPCC) and append the log residual
energy (the *source*). A unit test checks the result against a direct solver.

**8 kHz resampling?** The anti-alias low-pass at the new Nyquist (4 kHz) removes 4–8 kHz,
and upsampling cannot bring it back. w2v-BERT keeps 92 %; LPCC drops to 14 %.

**μ-law vs uniform?** μ-law compresses with log(1+μ|x|) before quantising, so quiet parts
get finer steps and SQNR stays roughly constant with level. That gives 95 % vs 84 % at
8 bits. Uniform SQNR ≈ 6.02b + 1.76 dB (unit-tested).

**EER / AUROC?** EER is the error rate at the threshold where false alarms = misses. AUROC
is the probability that an unknown clip scores higher than a known one (0.5 = chance).

## The model and the results
**Why layer 4?** Accuracy is 49 % at the input projection, about 97 % from layer 3 to 23,
and 78 % at the last layer. Layer 4 is the cheapest layer at peak accuracy (4 of 24 blocks).

**Generator or speaker?** With unseen test speakers, w2v-BERT drops only 97.6 → 97.0 %,
while MFCC drops 72.2 → 67.6 % and LPCC 71.2 → 62.9 %. Also, the demo's nearest neighbours
are usually *other speakers* using the same generator.

**GMM vs kNN?** A GMM on frame-level LFCC gets 96.6 %, because it models the whole frame
distribution. But it must be retrained (EM) for every new generator; kNN only stores
vectors.

**Unseen generators?** Support = A01–A06; the score is the mean distance to the 21 nearest
known clips. AUROC is 96.6 % for w2v-BERT vs 77.9 % for MFCC.

**A16 / A19?** They are the same algorithms as A04 / A06, renamed in the eval set. Forced to
choose a known generator, 100 % of A16 clips go to A04 and 98 % of A19 clips to A06. The
method matches the *algorithm*.

**What breaks it?** Additive noise (52 % at 20 dB, 29 % at 10 dB) and audio with only the
4–8 kHz band (chance).

**How did you avoid tuning on the test set?** Layer 4 and k = 21 come from the paper, and
the splits are fixed seeds. Nothing was chosen by looking at test accuracy.

## Limitations (say them before you are asked)
* A 9,800-clip subset of one corpus (ASVspoof 2019 LA).
* It assumes the clip is fake. Without a "human" class, a real clip lands next to A19.
* Open set across corpora is hard: 2025 TTS clips got weak votes but were not always flagged.
* The robustness test used a clean support set only; adding noisy examples would help.
