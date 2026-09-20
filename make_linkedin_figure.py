"""Two-panel summary figure for the write-up / LinkedIn post.

Left: accuracy change vs English on terminated answers, one dot per condition
(16 natural languages + notations), one row per model.  Right: share of answers
that never terminated in the constructed languages, grouped by model.

Reads the JSON that the analysis snippet in the session wrote (per-model deltas
and runaway rates); regenerate it from results_prompt_v2/ (prompt v2) and
results/ (GLM, prompt v1) with analyze.paired_vs_baseline(drop_truncated=True).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data = json.load(open(sys.argv[1]))
out = sys.argv[2]

MODELS = [  # display name, key, colour (validated categorical palette, fixed order)
    ("GPT-5.6 Luna (reasoning on)", "gpt-5.6-luna-think", "#2a78d6"),
    ("DeepSeek V4 Flash", "deepseek-v4-flash", "#eb6834"),
    ("Claude Haiku 4.5", "claude-haiku", "#1baf7a"),
    ("GLM-5.3", "tokenrouter", "#eda100"),
    ("gpt-4.1-mini", "gpt-4.1-mini", "#e87ba4"),
]
CONDS = [("English", "english"), ("Esperanto", "esperanto"), ("Toki Pona", "toki_pona"),
         ("Ithkuil", "ithkuil"), ("Lojban", "lojban")]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.4), dpi=150, gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.32})
fig.suptitle("Does the language of the chain of thought change the answer?  Mostly no — but some languages stop the model from finishing.",
             x=0.04, ha="left", fontsize=14.5, fontweight="bold", y=0.975)
fig.text(0.04, 0.925, "9 LegalBench classification tasks, 1,048 items per cell, paired against English on identical items, temperature 0, no output cap.",
         fontsize=10.5, color=INK2)

# ── Left: accuracy delta strip ─────────────────────────────────────────
ax1.set_title("Accuracy vs English, 16 natural languages & notations\n(answers that terminated)", loc="left", fontsize=12, pad=12)
ax1.axvline(0, color=INK2, lw=1)
ax1.axvspan(-3, 3, color=GRID, alpha=0.6, lw=0)
ax1.text(0, len(MODELS) - 0.45, "±3 points", ha="center", va="bottom", fontsize=9, color=INK2)
for i, (name, key, col) in enumerate(MODELS):
    y = len(MODELS) - 1 - i
    d = data[key]["deltas"]
    xs = list(d.values())
    ax1.scatter(xs, [y] * len(xs), s=70, color=col, edgecolor=SURF, linewidth=1.2, zorder=3)
    ax1.text(-10.9, y, name, ha="right", va="center", fontsize=11)
    worst = min(d, key=d.get)
    if d[worst] < -3.5:
        ax1.annotate(f"{worst.replace('_', ' ')} {d[worst]:+.1f}", (d[worst], y), (d[worst] - 0.3, y + 0.33),
                     fontsize=9, color=INK2, ha="left")
ax1.set_xlim(-10.5, 4.5); ax1.set_ylim(-0.7, len(MODELS) - 0.2)
ax1.set_yticks([]); ax1.set_xlabel("accuracy points vs English")
for s in ("top", "right", "left"): ax1.spines[s].set_visible(False)
ax1.grid(axis="x", color=GRID, lw=0.8); ax1.set_axisbelow(True)
ax1.text(-10.5, -0.62, "Each dot is one language or notation. Nothing beats English on any model;\nonly the oldest model (gpt-4.1-mini) loses more than 3 points.",
         fontsize=9.5, color=INK2, va="bottom")

# ── Right: runaway bars ────────────────────────────────────────────────
ax2.set_title("Answers that never terminated, by reasoning language\n(model loops until its token limit)", loc="left", fontsize=12, pad=12)
n = len(MODELS); w = 0.15
for i, (name, key, col) in enumerate(MODELS):
    ra = data[key]["runaway"]
    for j, (clabel, ckey) in enumerate(CONDS):
        x = j + (i - (n - 1) / 2) * (w + 0.012)
        if ckey not in ra:
            ax2.text(x, 1.5, "not\nrun", ha="center", va="bottom", fontsize=7, color=INK2)
            continue
        v, cnt = ra[ckey]
        ax2.bar(x, v, width=w, color=col, lw=0, zorder=3)
        lab = f"{v:.0f}%" if v >= 1 else "0"
        if key == "gpt-5.6-luna-think" and ckey != "english":
            lab = "0*"
        ax2.text(x, v + 1.2, lab, ha="center", va="bottom", fontsize=8.5, color=INK2, zorder=4,
                 bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none", alpha=0.85))
ax2.set_xticks(range(len(CONDS))); ax2.set_xticklabels([c for c, _ in CONDS], fontsize=11)
ax2.set_ylim(0, 105); ax2.set_ylabel("% of answers"); ax2.set_yticks([0, 25, 50, 75, 100])
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.grid(axis="y", color=GRID, lw=0.8); ax2.set_axisbelow(True)
ax2.legend(handles=[Line2D([], [], marker="s", ls="", color=c, markersize=10, label=nme) for nme, _, c in MODELS],
           loc="upper left", frameon=False, fontsize=10, ncol=1, handletextpad=0.4)
fig.text(0.52, 0.075, "* Luna's hidden reasoning is on; its visible text is a short write-up in the requested language and never loops.\n"
         "Haiku: Ithkuil n=33, Toki Pona n=89.  gpt-4.1-mini: n=90\u2013109 on Toki Pona, Ithkuil, Lojban.  All other bars n=1,048.\n"
         "Answers that did terminate scored within a few points of English in every language.",
         fontsize=8.8, color=INK2, va="top")
fig.text(0.04, 0.012, "github.com/hordruma/MultiLingual-Reasoning", fontsize=10, color=INK2)
fig.subplots_adjust(left=0.15, right=0.985, top=0.83, bottom=0.17)
fig.savefig(out, facecolor=SURF)
print("wrote", out)
