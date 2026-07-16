"""Regenerate the central figure and the bin table from the RAW traces in this repo.

    python3 code/make_figure.py          # needs only matplotlib/numpy

Reads  data/raw_occ_traces.csv  — one row per (episode, scan): the occ-IoU of every scan of every
episode of every condition (no_mem / perfect_mem / icl / notepad ep0 / notepad ep4; runs identified by
training-job ID).
Writes data/occ_by_scan_bin.csv  (bin means, per-run min/max, episode-bootstrap 95% CIs, sample sizes),
       assets/occ_by_scan_bin.png (the five-bar figure in the write-up, with title and annotation),
and    paper/occ_by_scan_bin.png  (the same figure styled for the paper: no title, larger fonts).

Nothing in the figure is hand-entered: every bar is an average over the raw rows in the CSV, so any
number in the write-up can be checked against this script's output. The bootstrap resamples EPISODES
(cluster bootstrap, 2000 draws, fixed seed) — scans within an episode are correlated, so resampling
scans would be too optimistic. For the trained condition the CI treats the 4 replicate runs' episodes
as one pool; the run-to-run spread is reported separately as the min-max whisker.
"""
import csv
import os
import random
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
LABELS = ["scans 1–5", "6–10", "11–15", "16–20", "21–25", "26–30"]
N_BOOT = 2000
SEED = 0

# ---- load: episodes[condition] = list of {scan: occ} dicts (one per episode) ----
episodes = defaultdict(dict)   # (condition, run, epoch) -> {episode_idx: {scan: occ}}
with open(os.path.join(ROOT, "data", "raw_occ_traces.csv")) as fh:
    for r in csv.DictReader(fh):
        key = (r["condition"], r["run"], r["epoch"])
        episodes[key].setdefault(int(r["episode"]), {})[int(r["scan"])] = float(r["occ"])

def eps(cond, epoch=None, run=None):
    """List of per-episode {scan: occ} dicts matching the filters."""
    out = []
    for (c, rn, ep), d in episodes.items():
        if c == cond and (epoch is None or ep == str(epoch)) and (run is None or rn == run):
            out.extend(d.values())
    return out

RUNS = sorted({rn for (c, rn, ep) in episodes if c == "notepad"})

def bin_means(ep_list):
    """Pooled mean occ per scan bin (every scan of every episode is one observation), plus counts."""
    means, ns = [], []
    for lo, hi in BINS:
        vals = [v for e in ep_list for s, v in e.items() if lo <= s <= hi]
        means.append(mean(vals) if vals else float("nan"))
        ns.append(len(vals))
    return means, ns

def bootstrap_ci(ep_list, n_boot=N_BOOT, seed=SEED):
    """Episode-level (cluster) bootstrap 95% CI of each bin mean."""
    rng = random.Random(seed)
    boots = [[] for _ in BINS]
    for _ in range(n_boot):
        sample = [ep_list[rng.randrange(len(ep_list))] for _ in range(len(ep_list))]
        for b, m in enumerate(bin_means(sample)[0]):
            boots[b].append(m)
    lo = [float(np.percentile(b, 2.5)) for b in boots]
    hi = [float(np.percentile(b, 97.5)) for b in boots]
    return lo, hi

conds = {
    "no_mem": eps("no_mem"),
    "perfect": eps("perfect_mem"),
    "icl": eps("icl", epoch=0),
    "np0": eps("notepad", epoch=0),
    "np4": eps("notepad", epoch=4),
}
stats = {k: bin_means(v) for k, v in conds.items()}
cis = {k: bootstrap_ci(v) for k, v in conds.items()}
run_means4 = {rn: bin_means(eps("notepad", epoch=4, run=rn))[0] for rn in RUNS}
lo4 = [min(run_means4[rn][b] for rn in RUNS) for b in range(len(BINS))]
hi4 = [max(run_means4[rn][b] for rn in RUNS) for b in range(len(BINS))]

nomem, perf, icl, np0, np4 = (stats[k][0] for k in ("no_mem", "perfect", "icl", "np0", "np4"))
n4 = stats["np4"][1]

print("bin        no-mem    ICL   np-ep0  np-ep4  [run min–max]     np-ep4 95% CI     n_ep4  perfect")
for i, lb in enumerate(LABELS):
    print(f"{lb:10} {nomem[i]:6.3f} {icl[i]:6.3f} {np0[i]:7.3f} {np4[i]:7.3f}  "
          f"[{lo4[i]:.3f}–{hi4[i]:.3f}]  [{cis['np4'][0][i]:.3f}–{cis['np4'][1][i]:.3f}]  {n4[i]:6d}  {perf[i]:6.3f}")

# ---- bin table ----
with open(os.path.join(ROOT, "data", "occ_by_scan_bin.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["scan_bin", "no_mem", "icl", "notepad_untrained_ep0", "notepad_trained_ep4",
                "perfect_mem",
                "notepad_trained_run_min", "notepad_trained_run_max", "n_trained_scans",
                "no_mem_ci95_lo", "no_mem_ci95_hi", "icl_ci95_lo", "icl_ci95_hi",
                "notepad_untrained_ci95_lo", "notepad_untrained_ci95_hi",
                "notepad_trained_ci95_lo", "notepad_trained_ci95_hi",
                "perfect_mem_ci95_lo", "perfect_mem_ci95_hi"])
    for i, lb in enumerate(LABELS):
        w.writerow([lb.replace("scans ", ""), f"{nomem[i]:.4f}", f"{icl[i]:.4f}",
                    f"{np0[i]:.4f}", f"{np4[i]:.4f}", f"{perf[i]:.4f}",
                    f"{lo4[i]:.4f}", f"{hi4[i]:.4f}", n4[i]]
                   + [f"{cis[k][s][i]:.4f}" for k in ("no_mem", "icl", "np0", "np4", "perfect")
                      for s in (0, 1)])
print("written data/occ_by_scan_bin.csv")

# ---- five-bar grouped chart: blog version (title + annotation) and paper version (clean) ----
COLORS = dict(nomem="#dcdcdc", icl="#bcbddc", np0="#c6dbef", np4="#a1d99b", perf="#8f8f8f")
VAL_COLORS = dict(np4="#1d7a34", icl="#756bb1", perf="0.25")

BLOG_LABELS = dict(
    nomem="no-mem — perfect scripted agent, reports only what it currently sees (floor: best possible without memory)",
    icl="ICL — untrained base, full scan history in the prompt, no notepad (g7dncu2c ep0)",
    np0="notepad-untrained — untrained base WITH the notepad tools (4 runs, ep0, pooled)",
    np4="notepad-trained — RL-trained notepad (same 4 runs, ep4; whisker = min–max of the 4 runs)",
    perf="perfect-mem — perfect scripted agent with total recall of every channel ever seen (ceiling: perfect memory)")
PAPER_LABELS = dict(
    nomem="no-memory floor (scripted oracle)",
    icl="ICL — full scan history in the prompt",
    np0="notepad — untrained",
    np4="notepad — RL-trained (whisker: min–max of 4 runs)",
    perf="perfect-memory ceiling (scripted oracle)")


def draw(path, paper=False):
    x = np.arange(len(LABELS))
    w = 0.16
    lab = PAPER_LABELS if paper else BLOG_LABELS
    fig, ax = plt.subplots(figsize=(10.0, 6.0) if paper else (12.0, 7.4))
    ax.bar(x - 2.0 * w, nomem, w, color=COLORS["nomem"], label=lab["nomem"])
    ax.bar(x - 1.0 * w, icl, w, color=COLORS["icl"], label=lab["icl"])
    ax.bar(x, np0, w, color=COLORS["np0"], label=lab["np0"])
    ax.bar(x + 1.0 * w, np4, w, color=COLORS["np4"],
           yerr=[np.subtract(np4, lo4), np.subtract(hi4, np4)], capsize=4,
           error_kw=dict(lw=1.3, ecolor=VAL_COLORS["np4"]), label=lab["np4"])
    ax.bar(x + 2.0 * w, perf, w, color=COLORS["perf"], label=lab["perf"])
    fs_val = 10 if paper else 8
    for i in range(len(LABELS)):
        ax.annotate(f"{np4[i]:.2f}", xy=(x[i] + 1.0 * w, hi4[i]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=fs_val,
                    color=VAL_COLORS["np4"], fontweight="bold")
        ax.annotate(f"{icl[i]:.2f}", xy=(x[i] - 1.0 * w, icl[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=fs_val, color=VAL_COLORS["icl"])
        ax.annotate(f"{perf[i]:.2f}", xy=(x[i] + 2.0 * w, perf[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=fs_val, color=VAL_COLORS["perf"])

    if not paper:
        ax.annotate("ICL peaks at scans 6–10, then collapses as the history\ngrows — by 26–30 it is barely above the no-memory floor",
                    xy=(5.0 - 1.0 * w, icl[5] * 0.6), xytext=(1.9, 1.015), fontsize=9.5,
                    color=VAL_COLORS["icl"],
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=VAL_COLORS["icl"], alpha=0.92),
                    arrowprops=dict(arrowstyle="->", color=VAL_COLORS["icl"], lw=1.2))
        ax.set_title("Can a trained notepad beat in-context history?  occ-IoU by scan position, qwen3-1.7b\n"
                     "no-memory floor · free in-context history (ICL) · untrained notepad · RL-trained notepad · perfect-memory ceiling")

    fs_label = 15 if paper else 10
    fs_tick = 14 if paper else 10
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.set_xlabel("scan position within the 30-scan episode", fontsize=fs_label)
    ax.set_ylabel("occupied-IoU" if paper else "occ-IoU  (higher = report closer to the true transmitter set)",
                  fontsize=fs_label)
    if paper:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=11.5, ncol=2,
                  framealpha=0.95)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), fontsize=8.2, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 1.06 if paper else 1.13)
    fig.tight_layout()
    fig.savefig(path, dpi=200 if paper else 150, bbox_inches="tight")
    plt.close(fig)


draw(os.path.join(ROOT, "assets", "occ_by_scan_bin.png"), paper=False)
print("written assets/occ_by_scan_bin.png")
draw(os.path.join(ROOT, "paper", "occ_by_scan_bin.png"), paper=True)
print("written paper/occ_by_scan_bin.png")
