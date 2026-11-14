# TADA presentation direction

## Goal
An evidence-led academic talk, not a marketing template. Preserve the project's measured results and course connections; separate observations from hypotheses.

## Visual system
- Canvas: 16:9, 13.333 × 7.5 inches.
- Light surface: `#FBFAF6`; primary ink: `#182B3A`; secondary ink: `#51616C`.
- Dark conclusion surface: `#122733`; light ink: `#FBFAF6`.
- Typography: Arial throughout; bold statement titles, regular body, italic qualifications. Commands use Arial too. One font family follows the academic presentation guide.
- Claim titles: 32–38 pt. Cover: 58 pt. Body: 20–24 pt. Figure captions: 14–16 pt. Sources: 10–12 pt.
- Margins: 0.65 inches. Consistent alignment; open whitespace; thin rules instead of rounded cards.
- Chart identity: blue `#2a78d6`, orange `#eb6834`. Validated against the light surface using the dataviz palette validator. Marker shape and labels also identify series.
- Evidence only: real project spectrograms and result JSON. No synthetic data graphics, stock photos, gradients, decorative icons or shadows.

## Story
1. Question: who made this voice?
2. Detection and attribution answer different questions.
3. Real spectra motivate generator fingerprints.
4. Frozen representations and course baselines feed the same k-NN decision.
5. Corpus and evaluation protocols.
6. Main attribution result.
7. High-band LFCC experiment and its limited interpretation.
8. Unseen-speaker test.
9. Open-set distance score, with protocol clearly stated.
10. Reused-generator controls.
11. Channel robustness and failure modes.
12. Demo commands and observable outputs.
13. Conclusion and honest limits.
14–17. Backup: layers, confusion matrices, full robustness table, references.

## Scientific guardrails
- 97.6% is five-split mean attribution on 13 eval generators; 97.0% is speaker-disjoint mean.
- 96.6% AUROC excludes the reused algorithms A16/A19 from the novel-generator aggregate.
- A16→A04 is 99.75%, not literally 100%; A19→A06 is 98.25%.
- Layer 4 is early and near peak, not the numerical best layer.
- The 4–8 kHz LFCC finding does not establish that SSL relies exclusively on that band.
- Human speech in the demo reference bank is not a separately evaluated real/fake detector.
- Keep partner placeholders visible; do not invent team identity or contributions.

## Quality gate
Render every slide through LibreOffice; inspect a contact sheet and all detailed pages. Check editable object bounds, paragraph fit, legibility, citations, aspect ratios and exact metric provenance. Keep original deck backed up before replacement; synchronize the submission copy after validation.
