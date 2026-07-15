# Run registry — every training run in this project

Selective reporting is the quiet failure mode of RL write-ups: train many, show the winner. This project
bought **43 training jobs**; the headline figure rests on **4** of them. This file lists *all* of them, in
approximately chronological order, including the failures, the cancelled run, the confounded run, and the
replication that did not reproduce — so the survivorship is auditable rather than taken on faith.

How to read the tables:

- **run ID** — the opaque training-job ID, the same ID used in the write-up and the data files. One row
  of `data/raw_occ_traces.csv` traces to one of these.
- **occ** — per-scan occupied-spectrum IoU; **early/late** — mean occ over scans 1–15 / 16–30;
  **a0 ≈ 0.21** — the scripted memoryless floor; **anchor** — mean occ over scans 1–2 (nearly memory-free).
- Every reward named below is implemented, verbatim, in [`code/spectrum_reward.py`](code/spectrum_reward.py).
- Unless a row says otherwise, main-series runs share one config: base **qwen3-1.7b**, the 24-row canonical
  band dataset, notepad memory mode, 12 candidates/group (GRPO), 8 epochs, lr 1e-4, LoRA rank 8,
  temperature 1.2, 8192-token turn cap. Each row states the **single lever** moved vs. its predecessor.

## Main series — agent-maintained notepad, 30-scan episodes

| run ID | arm | reward (what the model is paid for) | single lever vs. predecessor | outcome |
|---|---|---|---|---|
| `i7aw83iq` | dense-occ | mean occ over all 30 scans — "be accurate on every scan" | baseline of the series (procedural notepad prompt introduced) | ✗ Fixed notepad read-back (~91%) but occ flat across 8 epochs; policy drifted into verbatim copy-forward; merging never improved. |
| `b5h41zlu` | hinge | late − max(0, a0 − early) — pay late half, punish only sandbagging | stop paying early scans | ✗ **Confounded:** late +0.07 but early_dev rose in lockstep (+0.05) — the gain was *no-memory task skill*, which the goal forbids. |
| `jyiqys44` | abs v1 | late − \|early − a0\| — early half pinned to the floor from both sides | two-sided pin instead of one-sided hinge | ✗ **Cancelled at ep3 — design error caught mid-run:** scans 3–15 already carry real memory, so the pin taxed legitimate memory ("hide it until scan 16"). |
| `n4g52553` | abs v2 (anchor) | late − \|anchor − a0\| — only scans 1–2 pinned | pin window 1–15 → 1–2 | ✗ Pin worked (anchor_dev flat ≈0.03) but nothing climbed — with skill neutralized, this reward left no gradient the 1.7B could follow. |
| `pscoc9fp` | dormant | mean coverage of *recallable* channels (invisible now, seen earlier) − 3 exact-zero guards (anchor/area/width) | reward currency replaced: memory coverage instead of report accuracy | ✓ **Goal met:** late 0.44→0.54, held 6 epochs; scan-1 occ identical ep0 vs ep7 (no forbidden skill gain); merging up for the first time. |
| `g7dncu2c` | ICL twin *(comparison arm — ICL mode, no notepad; 4096-token cap; nudge dataset)* | mean occ over all 30 scans, context never wiped | memory mode: full-history ICL | ~ Trained (mean occ 0.344→0.363) but **late−early stayed negative every epoch**: it learned to *degrade less* over long context, never to accumulate. Its ep0 is the ICL bar in the central figure. |
| `jnlfncrh` | dormacq *(48-row dataset)* | dormant coverage + 0.5 × acquisition credit (channels first seen one scan ago) − same guards | + merge-event bonus term | ✗ **Paying for the merge event degraded it:** merge-OK 39.7→29.9%, copy-forward 58→68%, trained flat. Additive shaping term convicted. |
| `ge7q1hn4` | dorm48 *(48-row dataset)* | identical to `pscoc9fp` | dataset 24 → 48 rows (isolates `jnlfncrh`'s failure to the acq term) | ~ Found the basin (late 0.52 @ep4) then **fell out** at ep6; confirms the dormant-reward effect is real but fragile at this lr × row count. |
| `u38yq1gt` | dorm48-lo *(48 rows, lr 5e-5, 12 epochs)* | identical to `pscoc9fp` | lr 1e-4 → 5e-5 (+4 epochs) | ✗ Touched the basin twice, never held; ended below its own ep0. Low lr does not rescue retention. |
| `o4g4u90z` | rep2 | identical to `pscoc9fp` | **nothing** — field-for-field replication of `pscoc9fp` | ✗ **DID NOT REPRODUCE:** late 0.451→0.345, never entered the basin. Identical config, opposite trajectory → `pscoc9fp`'s hold was seed luck; run-to-run variance dominates. |
| `zk6w6fjn` | r4 (sweep 1/4) | identical to `pscoc9fp` | LoRA rank 8 → 4 | ✗ No hold; only arm still climbing at the end (0.495 at ep7) — within seed-variance band, not distinguishable from luck. |
| `mx9x3c7t` | r32 (sweep 2/4) | identical to `pscoc9fp` | LoRA rank 8 → 32 | ~ Best knob candidate: earliest basin entry (ep2), 5/8 epochs ≈0.48–0.50, but never a `pscoc9fp`-level hold. |
| `cgplwg2t` | lr5e5-24 (sweep 3/4) | identical to `pscoc9fp` | lr 1e-4 → 5e-5 on 24 rows | ✗ No basin; U-shape, ends at its own ep0 level. Consistent with `u38yq1gt`. |
| `pyl9pkw8` | lr2e4 (sweep 4/4) | identical to `pscoc9fp` | lr 1e-4 → 2e-4 | ⚠ **Split verdict:** largest genuine gain (complete-episode late 0.47→0.58, scan-1 flat) *contaminated* by a discovered exploit — the policy inflated its own output until episodes died at the token cap ~scan 24, and truncated episodes outscored complete ones. Motivated the completion guard. |
| `aesye5sz` `moh612qy` `upbrxew7` `w6swb23z` | scr1–4 *(8 candidates, 5 epochs — screening cadence)* | identical to `pscoc9fp` | 4 identical copies — a direct **seed-variance yardstick** | ✗ **0 of 4** basin entries in 5 epochs; endpoint spread 0.417–0.475. Establishes the noise band any knob effect must clear. |
| `qv2mk5k0` `xoi922eh` `t37x35oj` `aic2o1up` | dormc1–4 | dormant coverage − **4** guards: + completion hinge 1.5 × max(0, 30 − n − 2)/30 — "paid only for memory, and only if you finish" | + completion guard (kills `pyl9pkw8`'s truncation exploit); lr 2e-4 kept; screening cadence | ✓ **Memory-learning is repeatable:** all 4 scan-1-flat and exploit-free; complete-episode dormant coverage up in every run (+0.037/+0.122/+0.219/+0.267). Residual: 3 of 4 still truncated at the 8192 cap (taxed, not prevented) — the cap, not cheating, became the bottleneck. |
| `s8e07n53` `yp8deoer` `kym4znjc` `wqjyy66p` | **dormc16k-1–4 — the headline runs** | identical to dormc1–4 | turn cap 8192 → **16384** tokens | ✓ **The write-up's result:** all 4 scan-1-flat (+0.0004…+0.0011); dormant coverage up in every run (+0.041/+0.101/+0.181/+0.182); ep4 completion 100/97/60/90% (one run still truncates — disclosed as survivorship in the figure's `n` column). |

## Predecessor experiments — environment-echoed running list (before the agent-maintained notepad)

Earlier design in which the environment echoed a running list back into the prompt (memory maintenance was
scaffolded, not the agent's job). Superseded by the main series; reported because they were bought and
because their controls shaped the final design.

| run ID | arm | outcome |
|---|---|---|
| `btalo63n` | scaffold, full history in context (ICL-on) | memory_gain 0.075→0.126 — but confounded: the model could replay its in-prompt history. |
| `liabhmdn` | scaffold, context windowed off (notepad-echo only) | Small but real: +0.015 early→late rise. Replay, not the notepad, was the larger memory contributor. |
| `dmzj2mz8` | noise-reduction re-run (48 rows, lr 5e-5, 10 epochs) | The +0.015 rise **reproduced** with near-identical slope — trend real, magnitude small. |
| `q6lc11gu` | **scrambled-notepad causal control** (random freqs echoed, same structure) | Coverage collapsed to ≈ memoryless (0.149 vs 0.183 real) — the echoed *content*, not the format, carried the effect. |
| `tnfxdqkv` | probe (lr≈0) | Band-to-band noise floor: occ 0.452/0.474 on two fresh band sets → ±0.02 noise band. |
| `fva3tx6z` | echo study A — explicit prompt, real echo | occ 0.472 flat — saturated; nothing left to train. |
| `geote9qj` | echo study B — **scrambled control** | occ 0.356 flat — again content, not format. |
| `dtbn6lhm` | echo study C — weak prompt, real echo | occ 0.403→0.439 (slope +0.0037/ep). |
| `c4jk2e4z` | echo study C′ — replication of C | occ 0.401→0.425 (slope +0.0019/ep) — direction reproduced, magnitude halved. |

Three still-earlier exploratory jobs (`yysvs5nh`, `iotjbxqw`, `mb92hvps`) predate stable task plumbing and
are not comparable; they established one load-bearing diagnosis: the base model's reports contain **0%
hallucinated transmitters** — recalled channels are always real, so memory has no precision downside and
the problem was that nothing *paid* for it.

## Second task — database exploration (the honest failure)

Same base model, notepad memory, question-series episodes over an unknown database (see
[`task/task_description.md`](task/task_description.md), Task 2).

| run ID | arm | outcome |
|---|---|---|
| `w7guqqae` `xvmz0mxv` `qmi03m6j` `xmen1ghu` | 4 replicate GRPO runs | ✗ **Flat null, 4 of 4:** accuracy ~3%, informed-one-shot ≡ 0, across all epochs. |
| `v7uu671a` | pre-seed diagnostic — correct schema facts *handed* to the notepad at ep0 | ✗ Accuracy lifts only to 0.057 at ep0 and **decays under RL** (0.057→0.038) — an execution gap beneath the discovery gap. |

Plus one **local (non-training) sufficiency check**, n = 2 dataset rows: the same pre-seeded notepad given
to a large open model (gpt-oss-120b) through the identical evaluator path scored 0.93 and 0.67 (≈14/15 and
10/15 questions) — the facts suffice; the 1.7B cannot execute on them. Verdict: this task needs a larger
base model. Numbers in [`data/dbx_second_task.csv`](data/dbx_second_task.csv).

---

**Tally:** 43 training jobs — 26 main series (25 notepad arms + the ICL comparison arm), 9 predecessor,
3 early exploratory, 5 second task — plus one local, non-training sufficiency evaluation. Wins: the dormant-reward family trained repeatably (8 runs with the guarded reward, 8
scan-1-flat, dormant coverage up in all 8); everything else above is a documented failure, control, or
predecessor. The four headline runs were not selected *from* this list after the fact — they are the last
four runs of the design sequence the failures forced.
