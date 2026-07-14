# Task descriptions

Both tasks are **derived from** the Continual-Learning Bench (CLBench) and adapted into RL environments for
Fireworks RFT (GRPO). Below are the real prompts, the memory design, the scoring, and the anti-cheating
measures. (Benchmark canary strings from the original templates are intentionally omitted.)

---

## Task 1 — Blind spectrum monitoring *(the task that trains)*

### Setup
The agent monitors one fixed radio band containing ~11–14 transmitters (each ~8 MHz wide, in a 180 MHz
band). It receives **12 sequential scans of the same band**. In any given scan only ~3 transmitters are
active; the detector is noisy (miss probability 0.15, false-alarm probability 0.2). Dormant transmitters
still *occupy* the band — so to report all occupied regions the agent must **remember** transmitters seen in
earlier scans and keep reporting them while they're silent.

### The memory channel
Context is windowed to the current scan. After each report, the environment echoes the model's **own
previous report** back to it as a running list. That self-authored register is the *only* information that
crosses from one scan to the next — memory here is "what you chose to keep in your list."

### System prompt (the "explicit" variant, verbatim)
```
You are a spectrum monitoring analyst watching a radio-frequency band with multiple transmitter channels.
You receive a SERIES of scans of the SAME fixed band. Each scan lists noisy detected peaks (freq, power,
width). Only SOME transmitters are active in any given scan; others are temporarily dormant and reappear in
later scans. The full set of transmitters is unknown — you must INFER it by accumulating evidence ACROSS
scans.

Each scan, call submit_report with two equal-length lists, center_freqs and bandwidths (MHz), naming ALL
transmitter regions you believe persistently occupy the band — INCLUDING ones dormant in the current scan
but seen in earlier scans. Use the current peaks as new evidence, but report the full persistent set you
have inferred from the whole scan history so far, not just the current peaks. More complete and accurate
reports score higher.
```

The **weak** prompt used in the training arms (C/C′) describes the task and the running list but **removes**
the accumulation instruction (the "INCLUDING ones dormant …" paragraph) — so accumulation has to be *learned*
rather than *told*.

### An example scan turn
```
--- Scan 4/12 ---

Scan metadata:
  scan_id: s-4f1a
  timestamp_utc: 2026-05-02T14:07:33Z
  sensor_id: sensor-2
  detector_version: 3.1
  integration_time_ms: 200
  estimated_noise_floor_dbm: -98.4

Detected peaks:
  - peak_id: p0 | freq: 42.6 MHz | power: -71.2 dBm | width: 7.8 MHz
  - peak_id: p1 | freq: 118.9 MHz | power: -68.0 dBm | width: 8.1 MHz
  - peak_id: p2 | freq: 155.2 MHz | power: -80.5 dBm | width: 7.9 MHz
Band: 20-200 MHz

Submit your report.
```
The model then calls `submit_report(center_freqs=[...], bandwidths=[...])` naming **every** region it
believes is occupied — including transmitters last seen in scans 1–3 that produced no peak this time.

### Scoring
- **Per-scan reward** = `3 × occupied-IoU` between the reported regions and the true occupied set. A
  memoryless (current-scan-only) agent scores occ ≈ **0.16**; a naive accumulate-all agent ≈ **0.447**;
  an agent that also filters false alarms tops out around **0.50–0.55**.
- **`memory_gain`** (late-scan minus early-scan occ) is *measured but never rewarded* — rewarding a delta
  invites sandbagging.
- **carry-rate** (fraction of the echoed running list preserved into the next report) is read from traces
  as the cleanest behavioral memory signal.

### Arms
| arm | prompt | echoed running list |
|---|---|---|
| **A** | explicit | real (the model's own previous report) |
| **B** *(control)* | explicit | **scrambled** — random in-band frequencies, same count/format |
| **C, C′** | weak | real |

Arm B is the key control: a training effect that survives content-scrambling is *skill*, not *memory*.

---

## Task 2 — Database exploration *(the task that did not train on 1.7B)*

### Setup
The agent answers a series of natural-language questions about an **unknown** SQLite database, one question
at a time. Between questions the conversation is **wiped**; the notepad is the only thing that persists. Each
question allows a small budget of exploratory SQL queries before an answer is required.

### The memory currency — "informed one-shot"
The database has obfuscated table names and *encodings that differ between table groups* (prices in cents vs
dollars; timestamps as epoch-ms, epoch-s, or ISO text; verified flags as `0/1` vs `'true'/'false'`).
Discovering these facts costs queries. The memory payoff: discover a fact **once**, write it to the notepad,
and answer *later* questions in a single shot without re-querying. Credit for a one-shot answer requires an
**evidence gate** — the answered value must trace to a query result or a provenance-tagged notepad fact — so
the model can't just smuggle answers into the notepad.

### The procedural prompt (verbatim — teaches the *method*, never a variant's answers)
```
You are a database analyst answering a SERIES of questions about ONE unknown SQLite database, shown ONE AT A
TIME. You interact through the `act` tool.

Between questions your conversation is WIPED. The ONLY thing that persists is YOUR NOTEPAD, shown at the
start of every question. The database itself never changes — so anything you learn ONCE and write down, you
never rediscover.

The products are split into three groups of tables named items_g1 / items_g2 / items_g3 (with matching
fdbk_g1/g2/g3 for reviews, and sometimes attrs_gX / taxn_gX). Each group is ONE product category, but the
table names are obfuscated and the columns are ENCODED DIFFERENTLY between groups. Your job on the first
questions is to DISCOVER the layout and RECORD it; on later questions, ANSWER from the notepad in one shot.

=== BUILD THESE FACTS ONCE, STORE THEM, REUSE THEM ===

(A) MAP each group to its category. For each items_gX run:
      SELECT main_cat, COUNT(*) AS n FROM items_gX GROUP BY main_cat ORDER BY n DESC LIMIT 1
    The DOMINANT main_cat names the group. main_cat is NOISY — each group also holds a minority of other
    categories — so use it ONLY to name the group. When you ANSWER, aggregate over the WHOLE group table; do
    NOT filter on main_cat (that would drop legitimate rows and give a wrong count/average).

(B) NAIL the encodings per group — they differ between groups, so check EACH group:
    - PRICES (prc): Decimals like 16.99 => already dollars. Large round integers like 1699 => CENTS, use
      prc/100.0. If a prc_usd column exists, prefer it. Sanity: a typical price is single/double-digit
      dollars; if an average comes out ~1699 you left prices in cents.
    - TIMESTAMPS (ts): 13-digit integer => epoch MILLISECONDS (ts/1000); 10-digit => epoch SECONDS; text
      '2019-04-...' => ISO (year = substr(ts,1,4)).
    - VERIFIED (vrf): integer 0/1 => verified is vrf = 1; text => verified is vrf = 'true'.

(C) ANSWER FROM ONE FINAL AGGREGATE. Compute the answer with a single COUNT / AVG / MAX / SUM / MIN query
    over the whole group table (applying the unit conversion), and answer with the value FROM THAT RESULT.
    Never read a number off a raw sample row.

=== EACH QUESTION ===
1. Read your notepad. If it already has the group's table, units and any needed fact, go STRAIGHT to one
   final aggregate query — do NOT re-explore.
2. If facts are missing, discover them with a FEW targeted queries, then answer.
3. ANSWER with the exact value from your aggregate: a bare number or short text, nothing else.
4. ALWAYS pass notepad_update with the COMPLETE updated fact sheet — anything you leave out is forgotten
   forever.
```

### Scoring
Reward grades **efficiency at position ≥ 2** (later questions should be answered in fewer queries, if the
notepad is working), minus penalties for anchoring on the current question, failing to complete, and
unevidenced instant-correct answers. Reported accuracy is strict. See `../data/dbx_second_task.csv`.

### Anti-cheating design
1. **Six database variants** — permute the category→table mapping *and* the encoding-quirk assignments per
   dataset row, so schema facts can't be baked into the weights.
2. **Evidence gate** — one-shot credit only if the answered value appears in a query result or a
   provenance-tagged notepad fact.
3. **Bake penalty** — makes answer-smuggling negative-sum, not merely zero.
4. **Reward-line scrub** — the model's own text can't forge the environment's scoring lines.

### Result
Four replicate GRPO runs on Qwen3-1.7B: **flat null** (accuracy ~3%, one-shot ≡ 0). A pre-seed diagnostic
(hand the correct schema to the notepad) lifts accuracy only to 0.057 at epoch 0 and it decays under RL —
a **discovery gap and an execution gap beneath it**. The same pre-seeded notepad given to gpt-oss-120b scores
0.67–0.93, so the facts are answerable; the 1.7B simply can't execute them. This task needs a bigger base
model. See the blog post's "The second task didn't train" section.
