"""REFERENCE COPY — the scripted "no-mem" agent behind the grey bars in the central figure.

This is the exact code that produced the no_mem rows of data/raw_occ_traces.csv: a perfect
memoryless agent that, on every scan, reads the currently visible peaks out of the observation and
reports exactly those channels with their true center frequencies and widths (matched against the
hidden ground truth), then submits. It never remembers anything — so its occ-IoU is an UPPER BOUND
for any agent without memory: you cannot beat it by being "better at the task"; you can only beat
it by carrying information across scans.

It is NOT runnable from this repo alone: it drives the real task engine (`get_task_class(
"blind_spectrum_monitoring")`, `resolved_gt`, `occ_iou`, the 24-band canonical schedule), which is
derived from the Continual-Learning Bench and not redistributed here (see the attribution note in
the README). It is vendored so the "upper bound" claim is inspectable: the agent's only input per
scan is the current observation text (`query.prompt`), and the only thing it reports is the
channels whose peaks appear in that text. Its full per-scan output is in data/raw_occ_traces.csv
(condition = no_mem, 24 episodes x 30 scans), which code/make_figure.py averages into the grey bars.
"""
import json

# Engine imports (CLBench-derived task engine — not included in this repo):
#   from spectrum_adapter import band_seed
#   from bench_eval import load_default_schedule, resolved_gt, occ_iou, PEAK
#   from src.registry import get_task_class
#   from src.interface import Response
#   from src.tasks.blind_spectrum_monitoring.task import ScanReport, Transmitter
#
# PEAK is the regex that parses "freq: <f> MHz" peak lines out of the scan observation —
# i.e. the agent sees exactly what the model sees.


def memoryless_series(schedule_rows, _VARIANTS, band_seed, get_task_class, resolved_gt, occ_iou,
                      PEAK, Response, ScanReport, Transmitter):
    """One occ-IoU series (30 scans) per canonical band row — the perfect memoryless policy."""
    rows = [json.loads(l) for l in schedule_rows]
    row_ids = [r.get("input_metadata", {}).get("row_id") or r.get("id") for r in rows]
    out = []
    for rid in row_ids:
        variant = next((v for v in _VARIANTS if v in rid), "five_ch_wide")
        kwargs = dict(_VARIANTS[variant])
        kwargs["seed"] = band_seed(rid) % (2 ** 31 - 1)
        task = get_task_class("blind_spectrum_monitoring")(**kwargs)
        task.build_canonical_run_state()
        gt = resolved_gt(task, float(kwargs.get("W", 15.0)), float(kwargs.get("G", 9.0)))
        query = task.build_current_query()
        occs = []
        for _ in range(int(kwargs.get("num_instances", 30))):
            # 1) the ONLY input: peaks visible in the current observation text
            vis = set()
            for f_ in PEAK.findall(query.prompt):
                fv = float(f_)
                for gi, ch in enumerate(gt):
                    if abs(fv - ch["center_freq"]) <= ch["bandwidth"] / 2:
                        vis.add(gi)
            # 2) report exactly those channels, perfect widths — nothing recalled, nothing invented
            cfs = [gt[i]["center_freq"] for i in sorted(vis)]
            bws = [gt[i]["bandwidth"] for i in sorted(vis)]
            occs.append(occ_iou(cfs, bws, gt))
            sr = task.step(Response(action=ScanReport(transmitters=[
                Transmitter(center_freq=c, bandwidth=b, currently_active=True, estimated_power=-30.0)
                for c, b in zip(cfs, bws)]), metadata={}))
            if sr.done:
                break
            nq = getattr(sr, "next_query", None)
            if nq is not None:
                query = nq
        out.append(occs)
    return out
