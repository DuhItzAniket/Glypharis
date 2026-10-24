from __future__ import annotations

import json

import compute_ledger


def test_jobs_pair_by_experiment_even_when_they_overlap(tmp_path):
    gpu = tmp_path / "gpu.log"
    gpu.write_text(
        "2026-10-24 00:00:00 START train.py --config configs/rec/gw-rec-005.yaml --resume x\n"
        "2026-10-24 00:10:00 START scripts/rec_sanity.py --config configs/rec/gw-rec-020.yaml\n"
        "2026-10-24 00:20:00 FAIL  (exit 1) GW-REC-020 sanity :: OutOfMemoryError\n"
        "2026-10-24 01:00:00 DONE  GW-REC-005\n"
        "2026-10-24 01:00:01 START train.py --config configs/rec/gw-rec-006.yaml\n",
        encoding="utf-8",
    )
    cpu = tmp_path / "chain.log"
    cpu.write_text(
        "2026-10-24 00:00:00 START --experiment GW-BASE-003 --system s --task rec --eval-set v\n"
        "2026-10-24 03:00:00 DONE  --experiment GW-BASE-003 --system s --task rec --eval-set v\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert compute_ledger.main(["--gpu-logs", str(gpu), "--cpu-logs", str(cpu),
                                "--runs", str(tmp_path), "--out", str(out)]) == 0  # fmt: skip
    res = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    jobs = {(j["experiment"], j["kind"]): j for j in res["jobs"]}
    assert jobs[("GW-REC-005", "resume")]["hours"] == 1.0
    assert jobs[("GW-REC-020", "sanity")]["outcome"] == "failed (exit 1)"
    assert abs(jobs[("GW-REC-020", "sanity")]["hours"] - 1 / 6) < 1e-9
    assert jobs[("GW-REC-006", "train")]["outcome"] == "unfinished"
    assert res["totals"]["cpu_h"] == 3.0 and abs(res["totals"]["gpu_failed_h"] - 1 / 6) < 1e-9
    assert "| 07 |" in (out / "ledger.md").read_text(encoding="utf-8")
