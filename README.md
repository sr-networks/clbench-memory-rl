# Can RL train a small LLM to use a notepad better than its own context window?

A small, self-contained write-up of a reinforcement-learning result on **Qwen3-1.7B**. The
Continual-Learning Bench (CLBench) reported that an external **notepad** doesn't reach the accuracy of
plain **in-context learning** (ICL — keeping the whole history in the prompt). That's backwards from what
you'd hope, so we asked whether a small model can be **trained** to use a notepad well enough to match or
beat ICL — *without* getting any better at the underlying task. It can, and the reason ICL loses (it rots
over a long episode) is the interesting part. We also document, honestly, where the same recipe *fails*.

![occ-IoU by scan position](assets/occ_by_scan_bin.png)

**One-line result:** over a 30-scan continual-monitoring episode, in-context history wins only the first few
scans, then **collapses** as the transcript grows — by the end it's barely above a no-memory baseline. An
**untrained** notepad already beats ICL from scan 6 on; **RL-trained**, it climbs to ~0.61 occ-IoU and holds
— and it does so *without improving the single-step task* (scan-1 accuracy, where the notepad is empty, is
identical trained vs. untrained).

## What's in here

| path | what |
|---|---|
| **[blogpost.md](blogpost.md)** | The full write-up, with the central figure and the honesty caveats. |
| **[paper/paper.pdf](paper/paper.pdf)** | The same result as a short scientific paper (LaTeX source in [`paper/`](paper/)). |
| **[REGISTRY.md](REGISTRY.md)** | **Every training run this project bought — all 43, including every failure** — so the survivorship is auditable. |
| **[task/task_description.md](task/task_description.md)** | The CLBench-derived tasks — real prompts, the memory-only reward, and the anti-cheating design. |
| **[data/](data/)** | The numbers as CSV, **including the raw per-episode/per-scan traces** behind the figure. See the [data dictionary](data/README.md). |
| **[code/](code/)** | The executed reward code (verbatim), the scripted no-memory agent, and a script that regenerates the figure from the raw traces. |
| **[assets/](assets/)** | The figures. |

## The four conditions in the figure

| condition | what it is | memory | late-half occ (scans 16–30) |
|---|---|---|---|
| **no-mem** | scripted perfect agent — reports only what it currently sees, run through the real engine | none (upper bound) | ~0.27 |
| **ICL** | untrained base model, full scan history in the prompt, no notepad | in-context | **0.297** |
| **notepad-untrained** | the *same* untrained model, but with the notepad tools + windowed context | notepad | 0.454 |
| **notepad-trained** | that model after RL on a memory-only reward | notepad | **0.607** |

`no-mem` is not a strawman — it's the *upper bound* for any memoryless agent. ICL vs. notepad-untrained is a
pure memory-*mode* comparison (same weights); notepad-untrained vs. notepad-trained isolates the *training*
effect. Runs: Qwen3-1.7B, GRPO with LoRA adapters on a managed RL fine-tuning service. Notepad bars pool
four replicate runs (`s8e07n53`, `yp8deoer`, `kym4znjc`, `wqjyy66p`); ICL is job `g7dncu2c` (ep0).

## Why it's memory, not task skill

The reward pays for exactly one thing — reporting channels that are **invisible now but were seen earlier**
(orthogonal to what's on screen). An anchor guard pins the near-memory-free opening scans to the memoryless
floor. And on **scan 1**, where the notepad is empty, trained and untrained score identically (+0.0004 to
+0.0011 across four runs): the model got better at *carrying information forward*, not at the task. Guards
against blanket-report and weight-baking cheats round out the design (see the blog + task description).

## Honesty box

- **Survivorship:** at the trained epoch, episode completion across the four runs is 100/97/60/90% — one run
  truncates, so the late bins average only scans actually played (pooled n falls 3840 → 3171). Disclosed.
- **The ICL bar is the untrained base** (4096-token cap). We also trained an ICL arm; it learned to degrade
  a little less over long context but never learned to accumulate (late−early stayed negative).
- **This is our re-implementation of the CLBench task family**, not CLBench's exact harness — the point is
  the mechanism and the trainability, not a leaderboard number.
- **One model, one optimizer.** A second CLBench notepad task (`database_exploration`) is a **flat null** on
  the 1.7B even when handed the answer schema — a discovery gap and an execution gap beneath it. Documented,
  not hidden. See the blog's "Where it does not generalize" section and `data/dbx_second_task.csv`.

## Reproduce

The central figure is **recomputable from raw data in this repo**:

```
python3 code/make_figure.py     # matplotlib/numpy only
```

reads the raw per-episode, per-scan traces ([`data/raw_occ_traces.csv`](data/raw_occ_traces.csv), 54,420
rows — every scan of every episode of all four conditions) and regenerates
[`data/occ_by_scan_bin.csv`](data/occ_by_scan_bin.csv) (bin means, per-run min/max, episode-bootstrap 95%
CIs) and the figure PNG. Nothing in the figure is hand-entered. The reward code the training jobs executed
is vendored verbatim in [`code/spectrum_reward.py`](code/spectrum_reward.py) (with a standalone
self-check), and the scripted no-memory agent in [`code/memoryless_agent.py`](code/memoryless_agent.py).
Second-task numbers are in [`data/dbx_second_task.csv`](data/dbx_second_task.csv). The task engine and
training-harness glue are CLBench-derived and not redistributed — what *is* here is everything a rigged
experiment would have to hide: the scoring, the raw traces, and the [complete run history](REGISTRY.md).

## Attribution & license

Both tasks are **derived from** the Continual-Learning Bench (CLBench) task family (blind spectrum
monitoring; database exploration); this repo redistributes neither CLBench's data nor its canary strings —
only our own measured results and task re-descriptions. Repo content is released under the
[MIT License](LICENSE).
