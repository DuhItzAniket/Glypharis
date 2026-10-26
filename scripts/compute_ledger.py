"""Compute ledger: every GPU and CPU job the queues ran, finished or not, for the paper's compute
statement.

    python scripts/compute_ledger.py

Reads the queue logs, where each job has a START line and a DONE or FAIL line with timestamps:
``runs/gpu_queue.log`` (Phase 6), ``runs/phase07_queue.log`` (Phase 7) and
``data/processed/baselines/chain.log`` (Phase 5 baselines). A START is paired with the next end
line for the same experiment and kind (training, resume, sanity, resume check, baseline run), so
overlapping jobs still pair correctly. Wall-clock time is what the job occupied the device; the
training time in each run summary (``runs/<id>/summary.json``) is listed beside it. Jobs run by hand
outside a queue (the Phase 6 sanity sequence, the GPU resume check, crash probes, tests) are not in
these logs; the ledger says so instead of guessing them.

Writes ``reports/compute/ledger.md`` and ``ledger.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (START|DONE|FAIL)\s+(.*)$")
EXIT = re.compile(r"\(exit (\d+)\)")


@dataclass
class Job:
    device: str  # gpu or cpu
    experiment: str
    kind: str  # train, resume, sanity, resume_check, baseline
    detail: str
    start: str
    end: str | None
    hours: float | None
    outcome: str  # done, failed (exit N), unfinished


def phase_of(experiment: str) -> str:
    if experiment.startswith("GW-BASE"):
        return "05"
    n = int(experiment.rsplit("-", 1)[1]) if experiment.startswith("GW-REC") else -1
    return "06" if n in (1, 2) else "07" if n >= 3 else "?"


def classify(text: str, device: str) -> tuple[str, str, str]:
    """(experiment, kind, detail) for a log line's text."""
    if device == "cpu":
        # Flags ("--experiment GW-BASE-002 --system ... --task ... --eval-set ...") or, in the
        # first days, positional fields ("GW-BASE-002 ppocr_mobile rec hiertext_val").
        exp = re.search(r"(GW-BASE-\d{3})", text)
        detail = " ".join(re.findall(r"--(?:system|task|eval-set) (\S+)", text))
        if not detail and exp:
            detail = " ".join(text[exp.end() :].split("::")[0].split()[:3])
        return (exp.group(1) if exp else "?"), "baseline", detail
    exp = re.search(r"gw-rec-(\d{3})", text, re.IGNORECASE)
    experiment = f"GW-REC-{exp.group(1)}" if exp else "?"
    if "rec_resume_check" in text:
        return experiment, "resume_check", ""
    if "sanity" in text:
        return experiment, "sanity", ""
    return experiment, "resume" if "--resume" in text else "train", ""


def parse(log: Path, device: str) -> list[Job]:
    open_jobs: dict[tuple[str, str], tuple[str, str]] = {}
    jobs: list[Job] = []
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        stamp, what, text = m.groups()
        experiment, kind, detail = classify(text, device)
        key_kind = "train" if kind == "resume" else kind
        if device == "cpu":
            key_kind = detail
        if what == "START":
            if (
                experiment,
                key_kind,
            ) in open_jobs:  # killed by hand: no end line, so how long it ran is unknown
                start, start_kind = open_jobs.pop((experiment, key_kind))
                jobs.append(Job(device, experiment, start_kind, detail, start, stamp,
                                None, "interrupted"))  # fmt: skip
            open_jobs[(experiment, key_kind)] = (stamp, kind)
            continue
        if (experiment, key_kind) not in open_jobs:
            continue  # an end line without its start (log started mid-job)
        start, start_kind = open_jobs.pop((experiment, key_kind))
        code = EXIT.search(text)
        outcome = (
            "done" if what == "DONE" else f"failed (exit {code.group(1)})" if code else "failed"
        )
        jobs.append(Job(device, experiment, start_kind, detail, start, stamp,
                        hours_between(start, stamp), outcome))  # fmt: skip
    for (experiment, _), (start, kind) in open_jobs.items():
        jobs.append(Job(device, experiment, kind, "", start, None, None, "running"))
    return jobs


def hours_between(start: str, end: str) -> float:
    return (
        dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(start)
    ).total_seconds() / 3600


def finished_runs(runs: Path) -> set[str]:
    """Runs whose summary shows every planned epoch (or an early stop): their last training job
    finished even if its process then exited non-zero (the Phase 6 exit crash, fixed in c58c063)."""
    import yaml

    done = set()
    for summary in runs.glob("GW-REC-*/summary.json"):
        s = json.loads(summary.read_text(encoding="utf-8"))
        cfg = summary.parent / "config.resolved.yaml"
        epochs = (
            yaml.safe_load(cfg.read_text(encoding="utf-8"))["train"]["epochs"]
            if cfg.exists()
            else None
        )
        if s.get("stopped_early") or (epochs is not None and s.get("epochs_run", 0) >= epochs):
            done.add(s["experiment_id"])
    return done


def failed(job: Job) -> bool:
    return job.outcome.startswith("failed") or job.outcome == "interrupted"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    gpu_logs = [Path("runs/gpu_queue.log"), Path("runs/phase07_queue.log")]
    parser.add_argument("--gpu-logs", nargs="*", type=Path, default=gpu_logs)
    parser.add_argument("--cpu-logs", nargs="*", type=Path,
                        default=[Path("data/processed/baselines/chain.log")])  # fmt: skip
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("reports/compute"))
    args = parser.parse_args(argv)
    jobs = [j for log in args.gpu_logs if log.exists() for j in parse(log, "gpu")]
    jobs += [j for log in args.cpu_logs if log.exists() for j in parse(log, "cpu")]
    jobs.sort(key=lambda j: j.start)

    train_h: dict[str, float] = {}
    for summary in args.runs.glob("GW-REC-*/summary.json"):
        s = json.loads(summary.read_text(encoding="utf-8"))
        train_h[s["experiment_id"]] = s["train_seconds"] / 3600

    # A finished run's last training job counts as done even if the process exited non-zero.
    complete = finished_runs(args.runs)
    last_train: dict[str, Job] = {}
    for j in jobs:
        if j.kind in ("train", "resume"):
            last_train[j.experiment] = j
    for exp, j in last_train.items():
        if exp in complete and failed(j):
            j.outcome = f"done ({j.outcome} after the run finished)"

    def total(device: str, phase: str | None = None, only_failed: bool = False) -> float:
        return sum(j.hours or 0.0 for j in jobs if j.device == device
                   and (phase is None or phase_of(j.experiment) == phase)
                   and (not only_failed or failed(j)))  # fmt: skip

    per_exp: dict[str, dict[str, Any]] = {}
    for j in jobs:
        fresh = {"device": j.device, "phase": phase_of(j.experiment), "jobs": 0,
                 "failed_jobs": 0, "wall_h": 0.0}  # fmt: skip
        e = per_exp.setdefault(j.experiment, fresh)
        e["jobs"] += 1
        e["failed_jobs"] += failed(j)
        e["wall_h"] += j.hours or 0.0
    for exp, e in per_exp.items():
        e["train_h"] = train_h.get(exp)

    phases = sorted({phase_of(j.experiment) for j in jobs})
    rows = []
    for exp, e in sorted(per_exp.items()):
        trained = "" if e["train_h"] is None else f"{e['train_h']:.2f}"
        rows.append(f"| {exp} | {e['phase']} | {e['device'].upper()} | {e['jobs']} | "
                    f"{e['failed_jobs']} | {e['wall_h']:.2f} | {trained} |")  # fmt: skip
    md = [
        "# Compute ledger",
        "",
        "Generated by `scripts/compute_ledger.py` from the queue logs; do not edit by hand.",
        "",
        "Wall-clock hours each job occupied its device, finished or not. Jobs that did not finish",
        "ended in a crash, a planned stop, a failed sanity attempt or check, or a failed run",
        "(`research/failures.md`). They are not all waste: a training job that stopped resumed",
        "from its last checkpoint, so only the steps after it were repeated. A job killed",
        "by hand has no end line; its duration is unknown and adds nothing to the hours.",
        "",
        "| Phase | GPU-h (all) | GPU-h (jobs that did not finish) | CPU-h (all) | "
        "CPU-h (jobs that did not finish) |",
        "|---|---|---|---|---|",
        *[f"| {p} | {total('gpu', p):.1f} | {total('gpu', p, True):.1f} | {total('cpu', p):.1f} | "
          f"{total('cpu', p, True):.1f} |" for p in phases],
        f"| all | {total('gpu'):.1f} | {total('gpu', None, True):.1f} | {total('cpu'):.1f} | "
        f"{total('cpu', None, True):.1f} |",
        "",
        "## By experiment",
        "",
        "| Experiment | Phase | Device | Jobs | Failed jobs | Wall-clock h | Training h |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## Not in these logs",
        "",
        "Jobs run by hand outside a queue: the Phase 6 sanity sequence (its own report gives the",
        "durations), the redesigned GPU resume check, the exit-crash probes, the GW-REC-001/002",
        "re-scores, and all test runs. They are short next to the totals above but not counted.",
        "",
    ]  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    totals = {"gpu_h": total("gpu"), "gpu_failed_h": total("gpu", None, True),
              "cpu_h": total("cpu"), "cpu_failed_h": total("cpu", None, True)}  # fmt: skip
    result = {"jobs": [asdict(j) for j in jobs], "experiments": per_exp, "totals": totals}
    (args.out / "ledger.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (args.out / "ledger.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print("\n".join(md[8 : 10 + len(phases) + 1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
