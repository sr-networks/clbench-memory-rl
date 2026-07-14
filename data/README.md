# Data

Every number in the blog post and both figures comes from these files. Per-epoch values were fetched from
the Fireworks RFT job metrics (`curves.average.*`); carry-rates were computed from the downloaded rollout
traces. Runs are identified by their Fireworks job IDs so each row is traceable to its source job.

## `spectrum_occ_by_epoch.csv`
Per-epoch training curves for the five spectrum runs (the left panel of `assets/memory_result.png`).

| column | meaning |
|---|---|
| `run` | Fireworks RFT job ID (`tnfxdqkv` probe, `fva3tx6z` A, `geote9qj` B, `dtbn6lhm` C, `c4jk2e4z` C′) |
| `arm` | A = explicit prompt / B = scrambled control / C, C′ = weak prompt / probe = lr≈0 base |
| `prompt` | `explicit` or `weak` |
| `echo` | memory channel: `real` (own previous report) or `scrambled` (random in-band freqs) |
| `epoch` | 0-indexed training epoch |
| `mean_occ` | mean occupied-IoU (the reward signal / task score) |
| `memory_gain` | late-scan minus early-scan occ (measured, never rewarded) |
| `score` | raw RFT reward (3 × occ terms) |
| `mean_avail` | availability of scored scans |

## `spectrum_results_summary.csv`
One row per arm with start/end endpoints. `carry_start`/`carry_end` are the **memory carry-rate** (fraction
of the echoed running list preserved into the next report), computed from traces — this is the headline
memory metric and is *not* in the raw job curves.

| column | meaning |
|---|---|
| `run`, `arm`, `prompt`, `echo` | as above; plus `(local)` reference rows for the oracle and memoryless floors |
| `occ_start`, `occ_end` | occupied-IoU at first/last epoch |
| `carry_start`, `carry_end` | memory carry-rate at first/last epoch (blank where not applicable) |
| `note` | one-line interpretation |

## `dbx_second_task.csv`
The `database_exploration` (second task) results — the flat null, the pre-seed learnability diagnostic, and
the sufficiency check (`assets/dbx_second_task.png`).

| column | meaning |
|---|---|
| `run` | Fireworks job ID: baseline nulls (`w7guqqae`/`xvmz0mxv`/`qmi03m6j`/`xmen1ghu`), pre-seed diagnostic (`v7uu671a`), or the local sufficiency smoke |
| `condition` | `empty_notepad (baseline null)` or `oracle_facts_preseeded` |
| `model` | `qwen3-1.7b` or `gpt-oss-120b` |
| `epoch` | training epoch (blank for the one-off sufficiency runs) |
| `acc` | answer accuracy |
| `one_shot` | informed-one-shot rate (correct + ≤1 query at position ≥2, with evidence) |
| `mean_eff` | mean graded efficiency |
| `note` | context |

### Reference values used in the figures (from the writeup, not tabulated per-epoch)
- Spectrum: memoryless floor occ ≈ 0.16; accumulate-all oracle occ 0.447 ± 0.042; explicit-prompt carry
  ceiling 0.967.
- dbx: null band max-ever acc 0.039; gpt-oss-120b sufficiency 0.67 & 0.93 on the same pre-seeded notepad.
