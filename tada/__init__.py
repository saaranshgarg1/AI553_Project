"""TADA study: training-free attribution of audio deepfakes (AI553 course project).

data.py      paths, ASVspoof 2019 LA system table, clip list, train/test splits
audio.py     audio loading + channel degradations (sampling, quantisation, noise, filtering, MP3)
features.py  MFCC, LFCC, LPCC (classical) and w2v-BERT 2.0 (self-supervised) features
models.py    kNN, GMM, kNN-distance OOD score, EER
"""
