# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

AI553 project presenters showing how an audio clip relates to labeled reference clips.

## Product Purpose

Choose bundled audio or upload an audio file and visually locate it in a two-dimensional reference map with its kNN neighbors.

## Capabilities and Constraints

The existing Python project uses frozen w2v-BERT 2.0 layer-4 embeddings and 21-neighbor voting over 13 eval generators plus human speech. Keep the dashboard small and explainable. Exclude recognized reference clips from their own neighbor search. Two-dimensional projection is illustrative; classification distances stay in the original embedding space. Do not invent accuracy guarantees or ground-truth labels for uploaded recordings.

## Evidence on Hand

`demo_clips/` contains 14 labeled reference examples, human speech, and modern synthetic speech. `data/manifest.csv` and `features/ssl.npy` provide the reference bank.

## Open Decisions

Implementation default: local-only Python browser app. No public deployment or third-party audio upload requested.
