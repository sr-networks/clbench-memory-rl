# Can you train *memory* into a small LLM with RL?

*A controlled study on Qwen3-1.7B: reinforcement learning can improve a small model's memory recall on a
continual-monitoring task — but only within the envelope that a single paragraph of instruction defines,
and only when memory maintenance isn't the model's own job.*

---

## The question

"Memory" is a slippery word for language models. When an RL run makes a model better at a task that
*requires* remembering things, three very different things could be happening:

1. **Knowledge** — the model memorized the task content into its weights. (Not memory; cheating, if the
   test content repeats.)
2. **Skill** — the model got better at the single-step mechanics, independent of memory.
3. **Memory use** — the model actually got better at *carrying information forward* and acting on it.

Only the third is "training memory." Most positive results conflate all three. This write-up is an attempt
to isolate #3 on a small model (Qwen3-1.7B) and measure it directly — and, just as importantly, to report
where the same recipe fails.

The headline, up front:

![Memory result](assets/memory_result.png)

> GRPO improves the model's **memory carry-rate** — the fraction of its own remembered state it preserves
> each step — from **~0.89 to ~0.94**, replicated across two independent runs, on task content that never
> repeats, with a scrambled-memory control that stays flat. But one paragraph of explicit instruction
> installs a higher level (carry 0.967) for free, and 12 epochs of RL don't catch it.

---

## The task: watching a band you can't fully see

The task is derived from the Continual-Learning Bench: **blind spectrum monitoring.** The agent watches a
radio-frequency band with ~11–14 transmitters. It receives a series of 12 scans of the *same* fixed band.
The catch: each scan only shows the ~3 transmitters that happen to be active right now (and it's noisy —
false alarms and missed detections). Dormant transmitters still occupy the band; they just aren't emitting
this instant.

So to report **all** the occupied regions, the agent has to *remember* transmitters it saw in earlier scans
and keep reporting them even while they're silent. A memoryless agent that reports only the current scan's
peaks scores about **0.16** (occupied-IoU). An agent that accumulates everything it has ever seen scores
about **0.447**. The gap between those is exactly the value of memory.

The memory channel is deliberately simple and auditable: after each scan, the environment echoes the model's
*own previous report* back to it ("YOUR RUNNING LIST: …"), and the context is windowed to the current scan.
That self-authored list is the **only** thing that survives from scan to scan. Memory here is literally "what
you chose to keep writing down."

*(Full prompts, an example scan, and the exact scoring are in [`task/task_description.md`](task/task_description.md).)*

---

## Closing the three doors

The whole point is to make sure a training gain can *only* be memory. Three devices:

**1. Never-repeating content ("epoch salt").** Every band is freshly seeded, and re-seeded *every epoch* —
each RFT epoch runs in its own process and rotates the entire band set at import time (verified in-cloud:
one distinct salt per epoch). No band is ever seen twice, so nothing can be memorized into the weights;
every epoch is automatically a fresh-band evaluation. *(This closed a real bug: earlier runs had unknowingly
reused the same 48 bands — which would have let "knowledge" masquerade as "memory.")*

**2. A scrambled-memory control arm.** An identical training run, except the echoed running-list's *content*
is replaced each scan with random in-band frequencies (same count, same format, same instruction). Anything
that still improves under scrambling is non-memory skill. Anything that needs the *real* content is memory.

**3. Behavioral instrumentation.** Beyond the score, we read the behavior straight from the traces:
**carry-rate** = the fraction of the echoed list the model preserves into its next report; report size; and
the within-group correlation between carry and reward. (Noise ordering, empirically: carry-rate is the
cleanest signal, then occ, then `memory_gain`.)

The reward is deliberately dumb: **3 × mean occupied-IoU.** We never reward the memory *delta* directly —
rewarding "improvement over time" invites the model to sandbag early scans.

---

## Result 1 — RL trains memory recall

From a **weak** prompt (one that describes the task and the running list but never says "accumulate"),
GRPO improved recall in both independent runs:

- **carry-rate 0.888 → 0.938** (run `dtbn6lhm`) and **0.900 → 0.935** (run `c4jk2e4z`) — both converging
  near 0.94, and the *worst* candidates improved most;
- report size grew (12.0 → 14.5 and 12.9 → 13.5) — the model keeps more of the band in mind;
- occupied-IoU rose with consistent, same-sign slopes (+0.0037/epoch, +0.0019/epoch; both runs end at their
  maximum epoch).

On never-repeating bands, with the scrambled-memory arm flat, that improvement can only be **policy-level
memory use**. The left panel of the figure shows the landscape: the scrambled control (B) barely moves, the
two weak-prompt arms (C, C′) climb, and the explicit arm (A) sits pinned at the top.

**How the reward actually drives it.** Inside a GRPO candidate group (same band, same scans), the 12
candidates differ behaviorally (sd of carry-rate ≈ 0.073), the reward ranks that behavior almost perfectly
(**corr(carry, reward) = +0.75 / +0.80**), and the policy moves up the paid gradient. The reward shape
works; the only slack in the chain is band/detector noise diluting how much of that behavior shows up in the
score — which is exactly why C and C′ differ in magnitude.

**Weights on the claim.** The score-level effect is *small* — about +0.01–0.02 occ, comparable to the ±0.02
band-noise floor. It clears that floor not by magnitude but by (a) the same-sign slope across two runs and
(b) the direct behavioral carry-rate measure. And `memory_gain` (the late−early occ delta) rose in C but not
in C′ — it's the noisiest instrument, so the claim rests on carry-rate and the occ slope, not on gain.

---

## Result 2 — one paragraph of instruction beats all of it

Swap the weak prompt for an **explicit** one — add a single paragraph that says *"dormant transmitters still
occupy the band; report every transmitter you have ever seen"* — and the untrained model immediately scores
occ **0.472**. That's above **every** RL-trained level reached anywhere in this project, and above the naive
accumulate-everything oracle (0.447): the instructed model already *curates* — it filters false alarms rather
than blindly accumulating. Its carry-rate is 0.967 with reports at the slot cap.

That's why Arm A is flat for 12 epochs: not "RL can't learn here," but **no headroom left** (the sliver that
remained, RL took: carry 0.967 → 0.972). The practical ordering at 1.7B is unambiguous:

> **instruction ≫ RL ≫ nothing.**

RL's role *below* the instruction ceiling is to converge toward it, slowly. If you can write the paragraph,
write the paragraph. RL earns its keep only where you can't — where the right behavior can't be named in
advance, or where you're pushing past what instruction alone reaches.

---

## Result 3 — RL preserves *cheap* memory but destroys *expensive* memory

There's a sharp asymmetry worth flagging. In everything above, the **environment** maintains the memory
channel — it echoes the running list back for free. Under that regime, 12 epochs of RL held memory behavior
rock-steady (Arm A) or improved it (C/C′).

In an earlier chapter of this project, memory was instead the **agent's own job**: explicit
`notepad_read` / `notepad_write` tools it had to choose to use. There, the *same* optimizer **eroded** the
prompt-installed behavior over 20 epochs — the memory gain decayed from +0.055 toward −0.008, occ sliding
back to the memoryless floor. When memory upkeep costs actions, the policy drifts to the simpler
current-scan-only strategy. And from a *vague* prompt the 1.7B never explored notepad use at all — zero
accumulating rollouts for GRPO to reinforce, across two rollout interfaces and with thinking on and off.

So: **memory *use* is trainable; memory *maintenance*, at this scale, is not** — RL actively removes it when
it's expensive.

---

## The second task didn't train — and that's part of the result

A single positive task is a thin result. So we built a second, deliberately different CLBench task to the
same standard: **`database_exploration`.** The agent answers a series of natural-language questions about an
*unknown* SQLite database. The memory currency here is an **informed one-shot**: discover the schema quirks
once (which obfuscated table is which category; whether prices are in cents or dollars; whether timestamps
are epoch-ms or ISO), write them into the notepad, and answer *later* questions in one shot without
re-querying. It came with six anti-baking database variants, an evidence gate against answer-smuggling, and
red-teamed reward floors.

It does **not** train on the 1.7B.

![The second task didn't train](assets/dbx_second_task.png)

Four replicate GRPO runs are a flat null: accuracy ~3%, informed-one-shot ≡ 0, no upward trend in any run.
To find out *why*, we ran a clean single-lever diagnostic (`v7uu671a`): pre-seed the notepad with the
**correct** oracle schema facts, removing discovery entirely, and read accuracy. If discovery were the only
wall, accuracy should jump.

It rises to just **0.057** at epoch 0 — above the null's max-ever 0.039, so discovery genuinely *is* part of
the wall — and then **decays back into the null band** under RL. Handed the schema outright, the 1.7B still
gets 94% of answers wrong, because it can't reliably execute aggregate-SQL-plus-unit-conversion even when
told exactly what to do. That the facts themselves are answerable is not in doubt: a competent model
(gpt-oss-120b) handed the *same* notepad scores **0.67–0.93**.

The verdict: a **discovery gap and an execution gap beneath it.** Removing discovery gives only a small,
non-trainable lift. This task needs a bigger base model — it is not a reward-design failure, and hiding it
would misrepresent what "RL trains memory" is worth on a 1.7B.

---

## What it adds up to

On a small model, with content memorization made impossible and skill drift controlled to zero:

- **Memory *use* is trainable by RL** — measured behaviorally, replicated — but slowly, and only within the
  envelope instruction defines. *If you can write the instruction, write the instruction.*
- **Memory *maintenance* is not trainable at this scale** — RL destroys it when it costs actions.
- **Reward design was not the bottleneck** (corr ≈ +0.8 with the target behavior). **Content isolation
  was** — without fresh-content salting per epoch, every apparent gain is confounded by repetition (our own
  earlier "positive" runs included).
- **Model scale is the ceiling on breadth** — a second, harder memory task simply doesn't move on 1.7B, even
  handed the answer schema.

## Limitations

- One model (1.7B), one optimizer (GRPO via Fireworks RFT); lr / temperature / group-size not swept.
- The score-level effect sits inside the ±0.02 noise floor; it clears the floor by slope consistency and by
  the behavioral carry-rate measure, not by magnitude.
- occ ceilings are noise-limited, not 1.0 (accumulate-all 0.447; + perfect false-alarm filtering 0.513;
  + center denoising 0.549). Read all effect sizes against that compressed range.
- Arm band sets are drawn independently per epoch (no epoch id for paired salts); the ±0.02 probe noise
  floor bounds the resulting comparison noise.
- `memory_gain` replicated in only one of two C-runs; the memory claim rests on carry-rate + the occ slope.
- The second task's null is shown on the 1.7B only; a bigger base (qwen3-8b) was provisioned but not pursued.

---

*Data behind every figure: [`data/`](data/). Task details: [`task/task_description.md`](task/task_description.md).
Runs are Fireworks RFT job IDs (`fva3tx6z`, `geote9qj`, `dtbn6lhm`, `c4jk2e4z`, `v7uu671a`, `w7guqqae` …),
cited so each number is traceable to its source job.*
