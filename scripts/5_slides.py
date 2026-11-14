#!/usr/bin/env python3
"""Rebuild the presentation from measured results. Run after slide_assets.py."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "slides" / "assets"
R = json.loads((ROOT / "results" / "results.json").read_text())
BG, INK, MUTED, GRID = "FBFAF6", "182B3A", "51616C", "DCE1E2"
DARK, LIGHT, BLUE, ORANGE = "122733", "FBFAF6", "2A78D6", "EB6834"
W, H = 13.333333, 7.5
P = Presentation()
P.slide_width, P.slide_height = Inches(W), Inches(H)
P.core_properties.title = "Who made this voice? — Training-free audio deepfake attribution"
P.core_properties.author = "Saaransh Garg and partner (details pending)"
P.core_properties.subject = "AI553 Speech & Audio Processing, IIT Ropar"


def rgb(value):
    return RGBColor.from_string(value)


def text(s, content, x, y, w, h, size=22, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, line in enumerate(str(content).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text, p.alignment = line, align
        p.font.name, p.font.size = "Arial", Pt(size)
        p.font.bold, p.font.color.rgb = bold, rgb(color)
        p.space_after = Pt(7)
        p.line_spacing = 1.1
    return box


def rect(s, x, y, w, h, color=GRID):
    shape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(color)
    shape.line.fill.background()
    return shape


def rule(s, x, y, w, color=GRID):
    rect(s, x, y, w, .012, color)


def image(s, name, x, y, w, h=None, alt=""):
    shape = s.shapes.add_picture(str(ASSETS / name), Inches(x), Inches(y), width=Inches(w),
                                 height=Inches(h) if h else None)
    shape._element.nvPicPr.cNvPr.set("descr", alt)
    return shape


def slide(section, title, source="Project results: results/results.json", dark=False, notes=""):
    s = P.slides.add_slide(P.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(DARK if dark else BG)
    ink, muted = (LIGHT, "C0CDD3") if dark else (INK, MUTED)
    text(s, section.upper(), .65, .35, 11.8, .28, 11, True, muted)
    if title:
        text(s, title, .65, .88, 12.0, 1.15, 32, True, ink)
    text(s, source, .65, 6.98, 11.3, .30, 11, color=muted)
    text(s, f"{len(P.slides):02d}", 12.13, 6.98, .55, .30, 11, color=muted, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def chart(s, categories, series, x, y, w, h, kind=XL_CHART_TYPE.BAR_CLUSTERED,
          maximum=100, labels=True, legend=True):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series:
        data.add_series(name, list(values))
    c = s.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
    c.has_title = False
    c.font.name, c.font.size = "Arial", Pt(16)
    c.font.color.rgb = rgb(INK)
    c.has_legend = legend and len(series) > 1
    if c.has_legend:
        c.legend.position = XL_LEGEND_POSITION.BOTTOM
        c.legend.include_in_layout = False
        c.legend.font.name, c.legend.font.size = "Arial", Pt(14)
        c.legend.font.color.rgb = rgb(INK)
    ca, va = c.category_axis, c.value_axis
    ca.tick_labels.font.size, va.tick_labels.font.size = Pt(15), Pt(13)
    ca.tick_labels.font.color.rgb = va.tick_labels.font.color.rgb = rgb(MUTED)
    ca.format.line.fill.background()
    va.format.line.fill.background()
    va.minimum_scale, va.maximum_scale, va.major_unit = 0, maximum, 20
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(GRID)
    va.major_gridlines.format.line.width = Pt(.5)
    va.tick_labels.number_format = '0"%"'
    if kind == XL_CHART_TYPE.BAR_CLUSTERED:
        ca.reverse_order = True
        c.plots[0].gap_width = 110
    for i, se in enumerate(c.series):
        color = [BLUE, ORANGE][i % 2]
        se.format.fill.solid()
        se.format.fill.fore_color.rgb = rgb(color)
        se.format.line.fill.background()
    if labels:
        plot = c.plots[0]
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.name, dl.font.size = "Arial", Pt(14)
        dl.font.color.rgb = rgb(INK)
        dl.number_format = '0.0"%"'
    return c


def mean(feature, protocol="random"):
    return np.mean(R["attribution"]["features"][feature][protocol]) * 100


def stat(s, value, label, x, y, w=3.7, dark=False, size=62):
    text(s, value, x, y, w, 1.02, size, True, LIGHT if dark else INK)
    text(s, label, x, y + 1.12, w, .85, 20, color="C0CDD3" if dark else MUTED)


# 01 — The research question is the cover, not a generic subject label.
s = slide("AI553 / speech & audio processing", "", "IIT Ropar · November 2026", dark=True,
          notes="We study TADA: attribution, not just detection. Can a frozen speech representation identify the generator without fitting a new model? This is a course-scale reproduction and analysis, not a new state-of-the-art claim.")
text(s, "Who made\nthis voice?", .7, 1.32, 7.2, 2.3, 62, True, LIGHT)
text(s, "Training-free attribution\nof audio deepfakes", .75, 4.00, 6.6, 1.1, 25, color="C0CDD3")
text(s, "Saaransh Garg · 2024AIB1011\n[Partner name · student ID]", .75, 5.62, 7, .82, 19, color=LIGHT)
image(s, "spectro_wavernn.png", 8.20, 1.30, 4.42, 4.6,
      "Real A10 WaveRNN spectrogram crop; time left to right, frequency bottom to top.")
text(s, "A10 / WaveRNN · real corpus excerpt", 8.20, 6.12, 4.45, .35, 13, color="C0CDD3")

# 02 — Distinguish the decision problems; anchor the exact question.
s = slide("Question / forensic decision", "A fake/real label does not identify the generator.",
          "Method context: Stan et al. (2025), TADA", notes="Detection is a binary gate. Attribution chooses a generator. Open-set attribution also needs to reject generators absent from the reference bank. Our research question concerns the last two, not a separately validated fake/real detector.")
for x, n, label, question in [(.75, "01", "DETECTION", "Is it fake?"),
                             (4.85, "02", "ATTRIBUTION", "Who made it?"),
                             (8.95, "03", "OPEN SET", "Have we seen it?")]:
    text(s, n, x, 2.28, 3.6, .55, 30, True, MUTED)
    text(s, label, x, 3.10, 3.6, .4, 15, True)
    text(s, question, x, 3.65, 3.6, .8, 28, True)
rule(s, .75, 4.77, 11.8)
text(s, "Can frozen speech features identify generators\nand flag unseen ones—without task-specific training?", .75, 5.12, 11.8, 1.05, 27)

# 03 — Real spectrograms, no decorative waveform.
s = slide("Motivation / spectral evidence", "Different synthesis chains leave different spectral detail.",
          "Project STFT excerpts · ASVspoof 2019 LA (Wang et al., 2020)",
          notes="These are illustrative excerpts, not matched utterances or a controlled causal test. Each is 2.5 seconds, STFT 512 samples with 128-sample hop; magnitudes normalized per clip, shown -80 to 0 dB. The course vocabulary—source-filter, phase reconstruction and neural vocoders—gives us testable feature choices.")
for x, name, head in [(.9, "spectro_human.png", "Human speech"),
                      (4.94, "spectro_wavernn.png", "A10 · WaveRNN"),
                      (8.98, "spectro_griffinlim.png", "A11 · Griffin–Lim")]:
    text(s, head, x, 2.15, 3.5, .42, 21, True)
    image(s, name, x, 2.85, 3.52, 2.57, f"{head}, real spectrogram: 0–2.5 s, 0–8 kHz, per-clip -80–0 dB.")
    text(s, "0                         time (s)                         2.5", x, 5.54, 3.55, .3, 11, color=MUTED)
text(s, "8 kHz", .18, 2.85, .64, .4, 11, color=MUTED)
text(s, "0", .54, 5.10, .25, .3, 11, color=MUTED)
text(s, "Illustrative clips, not a causal test.\nSpectral differences motivate the feature experiments.", .9, 6.02, 11.7, .67, 20)

# 04 — Editable method diagram.
s = slide("Method / frozen encoder + reference bank", "Both feature paths feed the same nearest-neighbour decision.",
          "Stan et al. (2025), TADA · Project implementation: tada/features.py, tada/models.py",
          notes="SSL means self-supervised learning. The TADA path takes hidden state 4 from frozen w2v-BERT 2.0 and averages it over time into 1024 numbers. Course baselines use mean and standard deviation of cepstral frames. k-NN uses k=21; known-class votes determine identity, and mean neighbour distance provides an open-set score. Frozen pretrained does not mean the encoder was never trained.")
text(s, "16 kHz\nmono clip", .78, 3.2, 2, 1, 25, True)
text(s, "→", 2.68, 3.47, .55, .55, 32, color=MUTED)
rect(s, 3.45, 2.29, 5.05, 1.52, "EDF2F4")
text(s, "Frozen w2v-BERT 2.0 · layer 4", 3.7, 2.50, 4.6, .42, 21, True)
text(s, "Time mean → 1,024-dimensional vector", 3.7, 3.13, 4.6, .48, 19)
rect(s, 3.45, 4.14, 5.05, 1.52, "EDF2F4")
text(s, "MFCC / LFCC / LPCC", 3.7, 4.37, 4.6, .42, 21, True)
text(s, "Frame mean + standard deviation", 3.7, 4.98, 4.6, .48, 19)
text(s, "→", 8.77, 3.47, .65, .55, 32, color=MUTED)
text(s, "21 nearest\nreference clips", 9.7, 2.73, 2.82, 1.15, 25, True)
text(s, "Votes → identity\nDistance → novelty", 9.7, 4.30, 2.82, 1.05, 22)
text(s, "No task-specific encoder training. New class = new reference clips.", .78, 6.13, 11.8, .5, 22)

# 05 — Dataset and protocols remain explicit.
s = slide("Data / ASVspoof 2019 logical access", "One corpus supports two distinct evaluation protocols.",
          "data/manifest.csv · Wang et al. (2020), ASVspoof 2019",
          notes="The corpus subset has 9800 clips, 8.865 hours and 107 speakers. Nineteen attack IDs are not nineteen unique algorithms: A16 and A19 reuse A04 and A06. Main attribution uses 5200 fake eval clips, 400 each for A07–A19. Open-set references come from A01–A06, with genuine novel A07–A19 excluding duplicate controls in the aggregate. Five seeded random splits and five speaker-disjoint splits are used for attribution. Robustness uses seed zero only.")
stat(s, "9,800", "clips · 8.9 hours", .78, 2.17)
stat(s, "19", "attack IDs · TTS + VC", 4.96, 2.17)
stat(s, "107", "speakers", 9.13, 2.17, 3.3)
rule(s, .78, 4.25, 11.8)
text(s, "ATTRIBUTION", .78, 4.63, 5.5, .35, 15, True)
text(s, "A07–A19 · 400 clips each\n5 random + 5 speaker-disjoint splits", .78, 5.21, 5.55, 1.05, 21)
text(s, "OPEN SET", 7.03, 4.63, 5.5, .35, 15, True)
text(s, "Reference: A01–A06\nNovel tests exclude reused A16/A19", 7.03, 5.21, 5.5, 1.05, 21)

# 06 — Native chart, editable data workbook.
s = slide("Result 01 / generator attribution", "Frozen speech features reach 97.6% attribution accuracy.",
          notes="Five-split means, 13 classes, 7.7% chance. Blue is random split, orange is held-out speakers. The rounded values are computed directly from results JSON. This is attribution among fake clips and does not measure real/fake detection. Full per-feature data are embedded in the native chart workbook.")
keys = ["mfcc", "lpcc", "lfcc_low", "lfcc", "lfcc_high", "ssl"]
chart(s, ["MFCC", "LPCC", "LFCC 0–4 kHz", "LFCC full band", "LFCC 4–8 kHz", "w2v-BERT L4"],
      [("Random split", [mean(k) for k in keys]), ("Unseen speakers", [mean(k, "speaker") for k in keys])],
      .72, 2.03, 8.50, 4.6, maximum=110)
stat(s, f"{mean('ssl'):.1f}%", "13 generator IDs\nk-NN · k = 21", 9.55, 2.64, 3.07, size=55)
text(s, "Chance: 7.7%", 9.55, 5.13, 3.05, .42, 20, True)
text(s, "Five-split means", 9.55, 5.79, 3.05, .5, 18, color=MUTED)

# 07 — Separate course finding from SSL mechanism claims.
s = slide("Result 02 / course feature ablation", "High-band LFCC outperforms full-band LFCC.",
          notes="Random-split mean attribution with the same k-NN. The 4–8 kHz LFCC representation reaches 84.3%, 7.9 points above full-band LFCC. Mel compression is a plausible interpretation, not a causal proof. SSL still works after low-pass filtering and fails on high-pass-only waveforms, so do not claim that the frozen encoder uses only the high band. A trained LFCC GMM reaches 96.6%, reported in backup notes rather than mixed into this training-free comparison.")
chart(s, ["MFCC", "LFCC 0–4 kHz", "LFCC full band", "LFCC 4–8 kHz"],
      [("Random-split mean", [mean(k) for k in ["mfcc", "lfcc_low", "lfcc", "lfcc_high"]])],
      .72, 2.30, 8.35, 3.78, legend=False)
stat(s, f"+{mean('lfcc_high')-mean('lfcc'):.1f}", "percentage points\nvs. full-band LFCC", 9.55, 2.6, 3.07, size=57)
text(s, "Mel scaling compresses\nhigh-frequency detail.", 9.55, 5.02, 3.07, 1.05, 21)
text(s, "Evidence for the cepstral baseline—not proof of the SSL mechanism.", .78, 6.31, 11.8, .4, 18, color=MUTED)

# 08 — Exact change in generalization, not categorical proof.
s = slide("Result 03 / speaker-disjoint test", "Attribution remains strong on unseen speakers.",
          notes="Five speaker-disjoint splits use GroupShuffleSplit. SSL drops by about 0.6 percentage points, compared with 4.6 for MFCC and 8.3 for LPCC. This limits the speaker-confound explanation within this dataset; it does not prove that all speaker information is absent from embeddings.")
text(s, "97.6%", .86, 2.55, 4.00, 1.3, 70, True)
text(s, "→", 5.22, 2.65, 1.05, .9, 52, color=MUTED)
text(s, "97.0%", 7.07, 2.55, 4.9, 1.3, 70, True)
text(s, "Random split", .86, 4.02, 4.6, .55, 24)
text(s, "Unseen speakers", 7.07, 4.02, 5.0, .55, 24)
rule(s, .86, 5.06, 11.60)
text(s, "Accuracy drop / percentage points", .86, 5.45, 5.1, .45, 21)
for x, name, key in [(6.45, "SSL", "ssl"), (8.62, "MFCC", "mfcc"), (10.75, "LPCC", "lpcc")]:
    text(s, name, x, 5.40, 1.8, .35, 15, True, MUTED)
    text(s, f"−{mean(key)-mean(key, 'speaker'):.1f}", x, 5.98, 1.8, .57, 30, True)

# 09 — Distribution exhibit, with non-duplicate aggregate stated.
s = slide("Result 04 / open-set attribution", "Neighbour distance separates genuinely novel generators.",
          notes="Reference bank A01–A06; open-set novelty is mean distance to 21 neighbours. Novel AUROC excludes A16/A19 because they reuse known algorithms. Density plot is illustrative stored samples, not the full data used to compute AUROC. AUROC measures ranking, not accuracy at a selected operational threshold. Cross-corpus novelty remains unvalidated.")
image(s, "ood_distance.png", .72, 2.11, 8.45, 4.34,
      "Normalized distribution of mean 21-neighbour distances for stored known and novel-generator samples; full-set AUROC is reported separately.")
stat(s, f"{R['ood']['ssl']['auroc']*100:.1f}%", "AUROC\nNovel-generator ranking", 9.50, 2.67, 3.1, size=55)
text(s, "MFCC: 77.9%", 9.50, 5.10, 3.1, .45, 21, True)
text(s, "A16/A19 excluded\nfrom novel aggregate.", 9.50, 5.83, 3.1, .72, 18, color=MUTED)

# 10 — Controls use exact percentages, not rounded perfect claims.
s = slide("Control / reused synthesis algorithms", "Reused systems map back to their original algorithms.",
          "Project forced-choice results · Algorithm mapping: Wang et al. (2020), Table 1",
          notes="A16 and A19 are eval attack IDs, but their algorithms reuse A04 and A06. Forced-choice support includes only A01–A06. A16 maps to A04 in 399 of 400 clips (99.75%), and A19 maps to A06 in 393 of 400 clips (98.25%). Their per-system novelty AUROC is much lower, about 77.7% and 72.7%. This is a useful natural control, not proof of identical recordings.")
for y, left, right, value, desc in [(2.47, "A16", "A04", "99.75%", "Waveform concatenation"),
                                  (4.62, "A19", "A06", "98.25%", "Spectral filtering")]:
    text(s, left, .85, y, 2, .75, 43, True)
    text(s, "→", 3.00, y + .06, 1, .75, 38, color=MUTED)
    text(s, right, 4.45, y, 2, .75, 43, True)
    text(s, value, 8.92, y, 3.7, .9, 49, True)
    text(s, desc, .85, y + .91, 7, .5, 21, color=MUTED)
    text(s, "forced-choice matches", 8.92, y + 1, 3.7, .5, 18, color=MUTED)
rule(s, .85, 4.18, 11.65)

# 11 — Robustness chart, one split clearly distinguished from five-split averages.
s = slide("Result 05 / channel robustness", "Compression is tolerable; additive noise breaks attribution.",
          notes="Only test audio is degraded; reference bank stays clean. These are seed-zero results, not five-split averages. SSL accuracy: clean98.0, telephone92.5, uniform8-bit83.6, mu-law95.4, noise20dB51.6, noise10dB29.2, lowpass95.5, highpass-only7.7, MP3 90.7. Mu-law preserves performance better than uniform quantization at the same bit depth. This suggests noisy support examples as future work; that remedy was not tested.")
rb = R["robustness"]["ssl"]
conditions = ["clean", "telephone", "quant8", "mulaw8", "mp3_32k", "noise20", "noise10"]
chart(s, ["Clean", "Telephone / 8 kHz", "Uniform / 8-bit", "μ-law / 8-bit", "MP3 / 32 kbps", "Noise / 20 dB", "Noise / 10 dB"],
      [("SSL attribution", [rb[k]*100 for k in conditions])], .72, 2.06, 8.65, 4.43,
      maximum=110, legend=False)
stat(s, "51.6%", "at 20 dB noise\nvs. 98.0% clean", 9.66, 2.68, 2.94, size=53)
text(s, "Clean reference bank.\nDegraded test audio.\nOne fixed split.", 9.66, 5.04, 2.94, 1.24, 20)

# 12 — Demo is honestly described, never a fabricated terminal screenshot.
s = slide("Demo / observable outputs", "The demo exposes votes, neighbours and novelty distance.",
          "Implementation: demo.py · Prepared inputs: demo_clips/", notes="Run the random clip first; self-match is removed from the bank. Show majority votes, neighbour speakers, and distance percentile. The demo bank includes 13 fake classes plus bona fide clips. Its threshold is the 95th percentile of bank self-neighbour distances; it is a heuristic, not an independently calibrated cross-corpus threshold. Human and modern TTS examples illustrate behaviour but are not validated benchmarks.")
for y, command, label, desc in [(2.25, "python demo.py --seed 0", "HELD-OUT CLIP", "Prediction + 21-NN votes\nNearest clips and speaker IDs"),
                               (3.78, "python demo.py --file demo_clips/human.flac", "HUMAN INPUT", "Bona fide clips exist\nin the demo reference bank"),
                               (5.32, "python demo.py --file demo_clips/modern_tts.flac", "MODERN TTS", "Inspect distance percentile\nDo not promise correct rejection")]:
    text(s, label, .84, y, 7.90, .32, 13, True, MUTED)
    text(s, command, .84, y + .47, 8.10, .60, 20, True)
    text(s, desc, 9.42, y + .12, 3.20, .94, 19)
    if y < 5:
        rule(s, .84, y + 1.27, 11.77)

# 13 — Main talk ends here. Backups do not become the final spoken conclusion.
s = slide("Conclusion / leave on screen for Q&A", "Strong attribution. Narrow evidence. Clear limits.",
          "AI assistance: Claude Code (Anthropic) · All measurements from project scripts", dark=True,
          notes="Stop here and invite questions. The core result is training-free task adaptation using a frozen pretrained model, not a never-trained system. Course experiments identify useful cepstral frequency detail; they do not fully explain SSL. Main limits: one older corpus, 9800-clip subset, additive noise, no separately evaluated real/fake gate and no validated cross-corpus novelty threshold. Partner identity and team contributions must be completed before submission.")
for x, value, label in [(.84, "97.6%", "Generator attribution"), (4.98, "97.0%", "Unseen speakers"), (9.13, "96.6%", "Novel-generator AUROC")]:
    stat(s, value, label, x, 2.47, 3.46, dark=True, size=53)
rule(s, .84, 4.59, 11.69, "536772")
text(s, "Frozen encoder + reference clips; no task-specific training.", .84, 5.05, 11.7, .53, 24, color=LIGHT)
text(s, "Limits: one corpus · noise sensitivity · unvalidated cross-corpus novelty\nDemo bona fide class ≠ separately evaluated real/fake detector", .84, 5.86, 11.7, .88, 19, color="C0CDD3")

# 14 — Native line chart and precise near-peak interpretation.
s = slide("Backup A / layer selection", "Layer 4 is early and near peak—not uniquely optimal.",
          notes="Hidden states numbered zero=input projection, one through24=transformer blocks. Layer4 random97.615%, speaker97.030%. Layer19 random97.923%, so layer4 is not the numerical optimum. Course-scale reproduction follows TADA layer4. Truncating the encoder to four blocks is not implemented in current extraction, so no measured runtime speedup is claimed. The LFCC-frame GMM (trained) averages96.615%; it is a different fitting requirement.")
chart(s, [str(i) for i in range(25)], [("Random split", np.array(R['attribution']['layers']['random'])*100),
                                   ("Unseen speakers", np.array(R['attribution']['layers']['speaker'])*100)],
      .72, 2.10, 8.65, 4.45, kind=XL_CHART_TYPE.LINE, labels=False)
text(s, "Chosen: layer 4", 9.68, 2.63, 2.95, .65, 24, True)
text(s, "Identity appears early.\nFinal layer loses accuracy.\nNo runtime saving measured.", 9.68, 3.61, 2.95, 1.71, 21)
text(s, f"Trained LFCC GMM\n{np.mean(R['attribution']['gmm_lfcc'])*100:.1f}% mean accuracy", 9.68, 5.57, 2.95, .94, 19, color=MUTED)

# 15 — Heatmap is an image because PowerPoint has no native matrix heatmap form.
s = slide("Backup B / error structure", "Most SSL errors are confined to a few generator pairs.",
          notes="Row-normalized confusion matrices, random split seed zero. Values differ from the five-split mean. Rows=true generator; columns=predicted generator. Both matrices share a 0–100% sequential scale. A14 and A15 each have perfect class recall in this split, consistent with distinguishing their different vocoders, but that is not a broad causal identification claim.")
image(s, "confusion.png", .75, 2.05, 11.83, 4.66,
      "Row-normalized seed-zero confusion matrices for frozen w2v-BERT layer4 and LFCC, with shared 0–100% sequential color scale; A07–A19 true/predicted classes.")

# 16 — Exact tabular backup provides accessible values, no color-only heatmap.
s = slide("Backup C / complete channel results", "SSL tolerates low-pass audio but fails on high-pass-only audio.",
          notes="Accuracy percentages, seed-zero split, clean support bank. The high-pass-only result is7.7% for SSL while lowpass4k is95.5%. Combined with high-band LFCC ablation, this means useful forensic information depends on feature extractor and signal context; it does not imply a universal high-frequency-only fingerprint.")
headers = ["Test channel", "MFCC", "LFCC", "LPCC", "SSL L4"]
rows = [("Clean", "clean"), ("Telephone / 8 kHz", "telephone"), ("Uniform / 8-bit", "quant8"),
        ("μ-law / 8-bit", "mulaw8"), ("MP3 / 32 kbps", "mp3_32k"),
        ("Low-pass / 4 kHz", "lp4k"), ("High-pass / 4 kHz", "hp4k"),
        ("White noise / 20 dB", "noise20"), ("White noise / 10 dB", "noise10")]
xs, widths = [.85, 5.75, 7.45, 9.15, 10.85], [4.45, 1.5, 1.5, 1.5, 1.6]
for x, w, head in zip(xs, widths, headers):
    text(s, head, x, 2.13, w, .42, 18, True)
rule(s, .85, 2.69, 11.70)
for i, (name, key) in enumerate(rows):
    y = 2.85 + i*.415
    text(s, name, xs[0], y, widths[0], .35, 17)
    for x, w, feature in zip(xs[1:], widths[1:], ["mfcc", "lfcc", "lpcc", "ssl"]):
        text(s, f"{R['robustness'][feature][key]*100:.1f}%", x, y, w, .35, 17, feature == "ssl")
text(s, "One fixed split · Clean reference bank · Chance: 7.7%", .85, 6.70, 11.6, .30, 13, color=MUTED)

# 17 — Full citations, not only paper names scattered in notes.
s = slide("References / sources & reproduction", "Sources and scripts make every claim traceable.",
          "Presentation redesign: academic-pptx + Anthropic PPTX guidance · See slides/README.md",
          notes="References support the method and corpus mapping, not the project's numerical results. Project measurements are stored in results/results.json. Partner metadata remains pending. To rebuild, run scripts/slide_assets.py then scripts/5_slides.py. Return to slide13 for Q&A.")
text(s, "METHOD", .84, 2.17, 2.4, .35, 14, True, MUTED)
text(s, "Stan et al. (2025). TADA: Training-free Attribution of Deepfake Audio.\nInterspeech 2025. arXiv:2506.05802.", .84, 2.76, 11.6, 1.03, 22)
text(s, "CORPUS & GENERATOR MAPPING", .84, 4.05, 8.0, .35, 14, True, MUTED)
text(s, "Wang et al. (2020). ASVspoof 2019: A large-scale public database\nof synthesized, converted and replayed speech.\nComputer Speech & Language, 64, 101114. arXiv:1911.01601.", .84, 4.61, 11.6, 1.38, 21)
text(s, "Measured results: results/results.json · Demo: demo.py · Rebuild: scripts/5_slides.py", .84, 6.36, 11.6, .42, 16, color=MUTED)


def save(output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        backup = ROOT / "_archive" / "slide_backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup.mkdir(parents=True)
        shutil.copy2(output, backup / output.name)
        print("Previous deck backed up:", backup / output.name)
    P.save(output)
    print(f"Saved {len(P.slides)} slides: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "slides" / "AI553_TADA_presentation.pptx")
    save(parser.parse_args().output)
