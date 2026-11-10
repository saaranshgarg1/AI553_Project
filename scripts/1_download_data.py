#!/usr/bin/env python3
"""Download a speaker-balanced subset of ASVspoof 2019 LA (Hugging Face mirror Bisher/ASVspoof_2019_LA).

The full corpus is 7.6 GB of parquet files made of 100-row "row groups". We only download
what we need:
  1. read the small metadata columns (speaker, system, file name) of the row groups,
  2. plan which rows to take: walk row groups in random order and accept a row only while its
     class needs clips and its speaker has not used up its share,
  3. read only the planned row groups and save the FLAC bytes as they are.

Result: 400 clips per class for train (A01-A06 + bona fide) and eval (A07-A19 + bona fide),
200 for dev. The three partitions have different speakers.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

import _path  # noqa: F401  (makes `tada` importable)
from tada.data import DATA, ROOT

REPO = "datasets/Bisher/ASVspoof_2019_LA/data"
PARTS = {"train": ("train", 400, None), "dev": ("validation", 200, None), "eval": ("test", 400, 150)}
fs = HfFileSystem()


def read_metadata(url, max_row_groups, rng):
    n = pq.ParquetFile(fs.open(url)).metadata.num_row_groups
    rgs = range(n) if max_row_groups is None else rng.choice(n, max_row_groups, replace=False)

    def one(rg):
        df = pq.ParquetFile(fs.open(url, block_size=1 << 16)).read_row_group(
            rg, columns=["speaker_id", "audio_file_name", "system_id"]).to_pandas()
        return df.assign(row_group=rg)

    with ThreadPoolExecutor(4) as ex:
        meta = pd.concat(ex.map(one, rgs), ignore_index=True)
    return meta.assign(label=meta.system_id.replace("-", "bonafide"))


def plan(meta, per_class, rng):
    cap = {c: math.ceil(per_class / n) for c, n in meta.groupby("label").speaker_id.nunique().items()}
    count, chosen = {}, []
    for rg in rng.permutation(meta.row_group.unique()):
        for i, r in meta[meta.row_group == rg].iterrows():
            if count.get(r.label, 0) < per_class and count.get((r.label, r.speaker_id), 0) < cap[r.label]:
                chosen.append(i)
                count[r.label] = count.get(r.label, 0) + 1
                count[(r.label, r.speaker_id)] = count.get((r.label, r.speaker_id), 0) + 1
    return meta.loc[chosen]


def main():
    rng = np.random.default_rng(0)
    rows = []
    for part, (split, per_class, max_rgs) in PARTS.items():
        url = f"{REPO}/{split}-00000-of-00001.parquet"
        chosen = plan(read_metadata(url, max_rgs, rng), per_class, rng)
        out = DATA / "asvspoof2019_la" / part
        out.mkdir(parents=True, exist_ok=True)
        wanted = set(chosen.audio_file_name)

        def fetch(rg):
            table = pq.ParquetFile(fs.open(url, block_size=1 << 23)).read_row_group(
                int(rg), columns=["audio_file_name", "audio"])
            for rec in table.to_pylist():
                if rec["audio_file_name"] in wanted:
                    (out / f"{rec['audio_file_name']}.flac").write_bytes(rec["audio"]["bytes"])

        with ThreadPoolExecutor(4) as ex:
            list(ex.map(fetch, chosen.row_group.unique()))
        print(f"{part}: {len(chosen)} clips from {chosen.row_group.nunique()} row groups")
        for r in chosen.itertuples():
            path = out / f"{r.audio_file_name}.flac"
            rows.append({"uid": r.audio_file_name, "partition": part, "label": r.label, "speaker": r.speaker_id,
                         "path": str(path.relative_to(ROOT)), "duration": round(sf.info(path).duration, 2)})
    pd.DataFrame(rows).to_csv(DATA / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
