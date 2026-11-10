# Disclosure of AI-tool use

**Tool:** Claude Code (Anthropic), model Claude Opus 5.5, used as a coding and writing
assistant on our laptop.

| Area | What the assistant did | What we did |
|---|---|---|
| Literature | Retrieved the TADA (arXiv 2506.05802) and ASVspoof 2019 (arXiv 1911.01601) papers to confirm details (layer 4, k = 21, A16 = A04, A19 = A06) | Read the papers; checked the claims in the report |
| Data | Wrote the script that downloads a speaker-balanced ASVspoof 2019 LA subset | Chose the dataset and the research questions |
| Code | Wrote `tada/` (features, models), the experiment and figure scripts, `demo.py` and the unit tests; fixed bugs in our earlier prototype (wrong w2v-BERT layer index, mislabelled plot legend, fake system names) | Reviewed and ran the code; [TEAM: describe your own changes] |
| Experiments | Ran feature extraction and experiments on our GPU | Interpreted the results; [TEAM] |
| Writing | Drafted the report and the slides from the experiment outputs | [TEAM: describe your revisions] |

No result was invented: every number in the report and the slides is written by
`code/scripts/3_experiments.py` to `results/results.json` and can be regenerated with
`code/run_all.sh`. We take full responsibility for the submission.

> Before submitting: replace the `[TEAM: ...]` entries with an honest description of what each of you did.
