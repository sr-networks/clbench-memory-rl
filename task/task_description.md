# Task descriptions

Both tasks are **derived from** the Continual-Learning Bench (CLBench) and adapted into RL environments
for GRPO fine-tuning. Below are the real prompts, the memory design, the scoring, and the anti-cheating
measures. (Benchmark canary strings from the original templates are intentionally omitted.)

---

## Task 1 — Blind spectrum monitoring *(the task that trains)*

### Setup
The agent monitors one fixed radio band containing ~11–14 transmitters (each ~8 MHz wide, in a ~180 MHz
band). It receives **30 sequential scans of the same band**. In any given scan only ~3 transmitters are
active; the detector is noisy (missed detections, false alarms). Dormant transmitters still *occupy* the
band — so to report all occupied regions the agent must **remember** transmitters seen in earlier scans and
keep reporting them while they're silent. The per-scan score is **occupied-IoU** (occ): the overlap between
the frequency ranges reported occupied and the true persistent transmitter set, 0 to 1.

### The two memory modes we compare
- **ICL (in-context learning):** nothing is wiped. Every past scan, report and result stays in the prompt,
  and the model re-reads its own history. There is no notepad. This is CLBench's strong baseline.
- **Notepad:** the context is windowed to the **current scan only**. The model is given `notepad_read` /
  `notepad_write` tools and a short (~4000-char) buffer it may overwrite each turn. That buffer is the
  **only** information that crosses from one scan to the next. Maintaining it is the *agent's own job* —
  it must choose to write. On this page, "memory" always means this notepad, never the model's weights or
  its in-conversation context.

Two scripted agents, run through the real task engine, bracket the comparison. The **no-memory floor** — a
perfect agent that reports exactly the currently-visible channels with perfect widths — scores occ ≈ **0.26**;
no memoryless policy can beat it. The **perfect-memory ceiling** — the identical agent with a persistent
seen-set, reporting every channel it has ever seen — climbs from ≈0.60 (scans 1–5; the unseen channels are
unknowable early) to ≈**0.99** by scans 26–30. Every model condition lives between these bounds.

### System prompt (notepad arm — paraphrase)
The notepad system prompt states the task (a series of scans of one fixed band; only some transmitters
visible per scan; dormant ones still occupy the band) and teaches the **procedure**: on every scan, read the
notepad, merge the current peaks into it, write back the complete running set of channels you've ever seen
(center frequency + bandwidth), and report that full persistent set — not just what's visible now. The ICL
arm gets the same task description but re-reads its full history instead of a notepad. *(Paraphrased rather
than quoted verbatim, and the benchmark's canary strings are omitted.)*

### An example scan turn
```
--- Scan 4/30 ---

Scan metadata:
  scan_id: s-4f1a
  sensor_id: sensor-2
  detector_version: 3.1
  estimated_noise_floor_dbm: -98.4

Detected peaks:
  - peak_id: p0 | freq: 42.6 MHz | power: -71.2 dBm | width: 7.8 MHz
  - peak_id: p1 | freq: 118.9 MHz | power: -68.0 dBm | width: 8.1 MHz
  - peak_id: p2 | freq: 155.2 MHz | power: -80.5 dBm | width: 7.9 MHz
Band: 20-200 MHz

Update your notepad, then submit your report.
```
The model reports **every** region it believes is occupied — including transmitters last seen in earlier
scans that produced no peak this time — using only what it kept in the notepad.

### The reward — paid only for memory
RL uses one number per episode, computed from the per-scan tool-result metrics (the executed code is
vendored verbatim in [`../code/spectrum_reward.py`](../code/spectrum_reward.py) —
`compute_spectrum_dormant_completion_reward`):

```
Score = 3 × ( mean_dorm − pen_anchor − pen_carpet − pen_wmax − pen_complete )

mean_dorm    = average SCAN_DORM over scans 2…n (scan 1 has no recallable channels)
               SCAN_DORM = Tversky overlap (α=β=1) between the report and the set of channels that are
               INVISIBLE this scan but were seen earlier (the "recallable" set). Currently-visible
               channels are ignored entirely — so the reward is orthogonal to single-step task skill.
pen_anchor   = max(0, anchor − a0 − 0.10)      anchor = mean occ over scans 1–2 (near-memory-free);
                                               a0 = scripted memoryless floor per band variant
                                               (0.2082 / 0.2141 / 0.2368)
pen_carpet   = 4.0 × max(0, mean_rarea − 1.15) rarea = reported area ÷ true occupied area
pen_wmax     = 1.0 × max(0, mean_wmax − 1.5)   wmax  = widest reported region ÷ widest true channel
pen_complete = 1.5 × max(0, 30 − n − 2)/30     n = scans completed; exact-zero for 28–30

Special case: 0 scans completed (first-turn format failure) → Score = −0.2 flat.
```

Every hinge is **exact-zero on honest play** (all four measured ≈0 at epoch 0 in every run), so on honest
rollouts the reward is literally `3 × mean recallable-coverage` — a memory-only currency. `memory_gain`
(late-scan minus early-scan occ) is *measured but never rewarded* — rewarding a delta invites sandbagging.

### Anti-cheating design
The guards were each built to defeat a cheat we simulated offline before training:

1. **Reward orthogonal to visible channels** — coverage is scored only on the *recallable* (invisible-now)
   set, so "get better at the current scan" earns nothing. Task skill and memory are separated by
   construction.
2. **Anchor pin** (`pen_anchor`) — scans 1–2 are held to the memoryless floor, so no-memory skill gains
   don't pay and can be taxed.
3. **Carpet + width guards** (`pen_carpet`, `pen_wmax`) — defeat "blanket the whole band" and the
   "behave honestly on scans 1–2, then blanket" (latecarpet) cheats; false positives are scored against
   never-seen space, which an honest memory policy never paints but a **weight-baked grid** must.
4. **Completion hinge** (`pen_complete`) — pays only for finishing all 30 scans, killing the
   truncate-after-banking-coverage exploit.
5. **Scan-1 baking detector** — on scan 1 the notepad is empty, so any trained-vs-untrained gain there would
   be band knowledge in the weights, not memory. Measured flat (+0.0004…+0.0011 across four runs).

### Conditions in the central figure
| condition | model | memory |
|---|---|---|
| **no-mem** | scripted perfect agent through the real engine | none (floor) |
| **ICL** | untrained base, full history in prompt (job `g7dncu2c`, ep0) | in-context |
| **notepad-untrained** | untrained base with notepad tools (4 runs at ep0, pooled) | notepad |
| **notepad-trained** | same 4 runs after RL (`s8e07n53`/`yp8deoer`/`kym4znjc`/`wqjyy66p`, ep4) | notepad |
| **perfect-mem** | same scripted agent with a persistent seen-set (total recall) | perfect (ceiling) |

See [`../data/occ_by_scan_bin.csv`](../data/occ_by_scan_bin.csv) for every number in the figure.

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
