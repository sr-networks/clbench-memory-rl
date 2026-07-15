"""Regenerate the central figure and the bin table from the RAW traces in this repo.

    python3 code/make_figure.py          # needs only matplotlib/numpy

Reads  data/raw_occ_traces.csv  — one row per (episode, scan): the occ-IoU of every scan of every
episode of every condition (no_mem / icl / notepad ep0 / notepad ep4; runs identified by training-job ID).
Writes data/occ_by_scan_bin.csv  (bin means, per-run min/max, episode-bootstrap 95% CIs, sample sizes)
and    assets/occ_by_scan_bin.png (the four-bar figure in the write-up).

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
    "icl": eps("icl", epoch=0),
    "np0": eps("notepad", epoch=0),
    "np4": eps("notepad", epoch=4),
}
stats = {k: bin_means(v) for k, v in conds.items()}
cis = {k: bootstrap_ci(v) for k, v in conds.items()}
run_means4 = {rn: bin_means(eps("notepad", epoch=4, run=rn))[0] for rn in RUNS}
lo4 = [min(run_means4[rn][b] for rn in RUNS) for b in range(len(BINS))]
hi4 = [max(run_means4[rn][b] for rn in RUNS) for b in range(len(BINS))]

nomem, icl, np0, np4 = (stats[k][0] for k in ("no_mem", "icl", "np0", "np4"))
n4 = stats["np4"][1]

print("bin        no-mem    ICL   np-ep0  np-ep4  [run min–max]     np-ep4 95% CI     n_ep4")
for i, lb in enumerate(LABELS):
    print(f"{lb:10} {nomem[i]:6.3f} {icl[i]:6.3f} {np0[i]:7.3f} {np4[i]:7.3f}  "
          f"[{lo4[i]:.3f}–{hi4[i]:.3f}]  [{cis['np4'][0][i]:.3f}–{cis['np4'][1][i]:.3f}]  {n4[i]:6d}")

# ---- bin table ----
with open(os.path.join(ROOT, "data", "occ_by_scan_bin.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["scan_bin", "no_mem", "icl", "notepad_untrained_ep0", "notepad_trained_ep4",
                "notepad_trained_run_min", "notepad_trained_run_max", "n_trained_scans",
                "no_mem_ci95_lo", "no_mem_ci95_hi", "icl_ci95_lo", "icl_ci95_hi",
                "notepad_untrained_ci95_lo", "notepad_untrained_ci95_hi",
                "notepad_trained_ci95_lo", "notepad_trained_ci95_hi"])
    for i, lb in enumerate(LABELS):
        w.writerow([lb.replace("scans ", ""), f"{nomem[i]:.4f}", f"{icl[i]:.4f}",
                    f"{np0[i]:.4f}", f"{np4[i]:.4f}", f"{lo4[i]:.4f}", f"{hi4[i]:.4f}", n4[i]]
                   + [f"{cis[k][s][i]:.4f}" for k in ("no_mem", "icl", "np0", "np4") for s in (0, 1)])
print("written data/occ_by_scan_bin.csv")

# ---- four-bar grouped chart ----
x = np.arange(len(LABELS))
w = 0.20
fig, ax = plt.subplots(figsize=(12.0, 7.4))
ax.bar(x - 1.5 * w, nomem, w, color="0.62",
       label="no-mem — perfect scripted agent, reports only what it currently sees (upper bound, no memory)")
ax.bar(x - 0.5 * w, icl, w, color="tab:purple",
       label="ICL — untrained base, full scan history in the prompt, no notepad (g7dncu2c ep0)")
ax.bar(x + 0.5 * w, np0, w, color="#9ecae1",
       label="notepad-untrained — untrained base WITH the notepad tools (4 runs, ep0, pooled)")
ax.bar(x + 1.5 * w, np4, w, color="tab:green",
       yerr=[np.subtract(np4, lo4), np.subtract(hi4, np4)], capsize=4,
       error_kw=dict(lw=1.3, ecolor="#14521c"),
       label="notepad-trained — RL-trained notepad (same 4 runs, ep4; whisker = min–max of the 4 runs)")
for i in range(len(LABELS)):
    ax.annotate(f"{np4[i]:.2f}", xy=(x[i] + 1.5 * w, hi4[i]), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=8, color="darkgreen", fontweight="bold")
    ax.annotate(f"{icl[i]:.2f}", xy=(x[i] - 0.5 * w, icl[i]), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=8, color="tab:purple")

ax.annotate("ICL peaks at scans 6–10, then collapses as the history\ngrows — by 26–30 it is barely above the no-memory bound",
            xy=(5.0 - 0.5 * w, icl[5] + 0.012), xytext=(2.75, 0.40), fontsize=9.5, color="tab:purple",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="tab:purple", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_xlabel("scan position within the 30-scan episode")
ax.set_ylabel("occ-IoU  (higher = report closer to the true transmitter set)")
ax.set_title("Can a trained notepad beat in-context history?  occ-IoU by scan position, qwen3-1.7b\n"
             "no-memory bound · free in-context history (ICL) · untrained notepad · RL-trained notepad")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), fontsize=8.6, framealpha=0.95)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, 0.78)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "assets", "occ_by_scan_bin.png"), dpi=150)
print("written assets/occ_by_scan_bin.png")
