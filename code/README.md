# Code

Three files, so the central claims can be checked rather than trusted:

| file | what | runnable? |
|---|---|---|
| [`make_figure.py`](make_figure.py) | Regenerates the central figure **and** `data/occ_by_scan_bin.csv` (bin means, min–max run whiskers, episode-bootstrap 95% CIs) from the raw traces in [`data/raw_occ_traces.csv`](../data/raw_occ_traces.csv). Nothing in the figure is hand-entered. | **yes** — `python3 code/make_figure.py` (matplotlib/numpy only) |
| [`spectrum_reward.py`](spectrum_reward.py) | The reward file the training jobs executed — every scoring line byte-identical to the executed original; only the harness's result types are replaced by dependency-free stand-ins. Contains the headline reward (`compute_spectrum_dormant_completion_reward`) **and** every earlier reward design, matching [`../REGISTRY.md`](../REGISTRY.md) row by row. | **yes** — `python3 code/spectrum_reward.py` runs a small self-check (honest / truncated / failed episode) |
| [`memoryless_agent.py`](memoryless_agent.py) | Reference copy of the scripted perfect no-memory agent (the grey "upper bound" bars). Its only per-scan input is the observation text; its full output is the `no_mem` rows of the raw-traces CSV. | no — it drives the CLBench-derived task engine, which is not redistributed here |

What is **not** vendored: the task engine, dataset builders and training-harness glue (CLBench-derived;
see the attribution note in the top-level README). The intent of this directory is that the *scoring* —
the part where a rigged experiment would hide — is fully inspectable, and the figure is fully
recomputable from raw data.
