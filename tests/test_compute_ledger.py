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
        "2026-10-24 03:00:00 DONE  --experiment GW-BASE-003 --system s --task rec --eval-set v\n"
        "2026-10-21 21:05:54 START GW-BASE-002 ppocr_mobile rec glyphsynth_val\n"
        "2026-10-21 21:31:54 DONE  GW-BASE-002 ppocr_mobile rec glyphsynth_val :: 10000 new\n",
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
    assert jobs[("GW-REC-006", "train")]["outcome"] == "running"  # still going: not a failure
    assert jobs[("GW-BASE-002", "baseline")]["detail"] == "ppocr_mobile rec glyphsynth_val"
    totals = res["totals"]
    assert abs(totals["cpu_h"] - (3 + 26 / 60)) < 1e-9
    assert abs(totals["gpu_failed_h"] - 1 / 6) < 1e-9
    assert "| 07 |" in (out / "ledger.md").read_text(encoding="utf-8")


def test_killed_jobs_and_a_finished_runs_exit_crash_are_accounted(tmp_path):
    gpu = tmp_path / "gpu.log"
    gpu.write_text(
        "2026-10-24 04:11:01 START train.py --config configs/rec/gw-rec-007.yaml --resume x\n"
        "2026-10-24 04:18:57 START train.py --config configs/rec/gw-rec-007.yaml --resume x\n"
        "2026-10-24 04:44:34 FAIL  (exit 127) GW-REC-007 :: exit crash\n",
        encoding="utf-8",
    )
    run = tmp_path / "runs" / "GW-REC-007"
    run.mkdir(parents=True)
    summary = {"experiment_id": "GW-REC-007", "epochs_run": 8, "stopped_early": False,
               "train_seconds": 3600}  # fmt: skip
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "config.resolved.yaml").write_text("train:\n  epochs: 8\n", encoding="utf-8")
    out = tmp_path / "out"
    assert compute_ledger.main(["--gpu-logs", str(gpu), "--cpu-logs", "--runs",
                                str(tmp_path / "runs"), "--out", str(out)]) == 0  # fmt: skip
    res = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    first, last = res["jobs"]
    assert first["outcome"] == "interrupted" and first["hours"] is None  # no end line: unknown
    assert last["outcome"].startswith("done")  # it finished the run, then exited non-zero
    assert res["experiments"]["GW-REC-007"]["failed_jobs"] == 1


def test_detector_chain_lines_pair_and_cpu_steps_are_skipped(tmp_path):
    gpu = tmp_path / "chain.log"
    gpu.write_text(
        "2026-10-26 21:11:55 START rec_profile GW-REC-024\n"
        "2026-10-26 21:12:06 DONE  rec_profile GW-REC-024 :: profile GW-REC-024: mean 24.8 ms\n"
        "2026-10-26 21:12:15 START det_sanity GW-DET-001 overfit lr_range dry_run\n"
        "2026-10-26 21:42:15 FAIL  det_sanity GW-DET-001 overfit lr_range dry_run :: lr_range\n"
        "2026-10-26 21:58:57 START det_resume_check GW-DET-001\n"
        "2026-10-26 22:04:57 DONE  det_resume_check GW-DET-001 :: identical\n"
        "2026-10-26 22:08:46 START train.py --config configs/det/gw-det-001.yaml\n"
        "2026-10-27 03:08:46 DONE  GW-DET-001\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert compute_ledger.main(["--gpu-logs", str(gpu), "--cpu-logs", "--runs", str(tmp_path),
                                "--out", str(out)]) == 0  # fmt: skip
    res = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    jobs = {(j["experiment"], j["kind"]): j for j in res["jobs"]}
    assert set(jobs) == {("GW-DET-001", "sanity"), ("GW-DET-001", "resume_check"),
                         ("GW-DET-001", "train")}  # fmt: skip
    assert jobs[("GW-DET-001", "train")]["hours"] == 5.0
    assert jobs[("GW-DET-001", "sanity")]["outcome"] == "failed"
    assert res["experiments"]["GW-DET-001"]["phase"] == "08"


def test_phase11_chain_lines(tmp_path):
    gpu = tmp_path / "chain.log"
    gpu.write_text(
        "2026-10-28 17:27:59 START P11 GLYPH-C score GW-REC-018\n"
        "2026-10-28 17:39:25 START P11 GLYPH-C score GW-REC-018\n"
        "2026-10-28 17:44:56 DONE  P11 GLYPH-C score GW-REC-018 :: mCE 0.1004\n"
        "2026-10-28 18:01:00 START P11 dry run GW-DATA-001 (shared training loop)\n"
        "2026-10-28 18:11:00 DONE  P11 dry run GW-DATA-001 (shared training loop) :: ok\n"
        "2026-10-28 20:00:00 START P11 train GW-DATA-003\n"
        "2026-10-28 21:00:00 FAIL  P11 train GW-DATA-003 :: stopped\n"
        "2026-10-28 22:00:00 START P11 train GW-DATA-003\n"
        "2026-10-28 22:30:00 DONE  P11 train GW-DATA-003 :: report\n"
        "2026-10-28 22:30:00 START P11 report GW-DATA-003\n"
        "2026-10-28 22:30:15 DONE  P11 report GW-DATA-003 :: report\n"
        "2026-10-29 04:12:58 START P11 compare the arms (GW-DATA-009)\n"
        "2026-10-29 04:13:16 DONE  P11 compare the arms (GW-DATA-009) :: H3\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert compute_ledger.main(["--gpu-logs", str(gpu), "--cpu-logs", "--runs", str(tmp_path),
                                "--out", str(out)]) == 0  # fmt: skip
    res = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    kinds = [(j["experiment"], j["kind"], j["outcome"]) for j in res["jobs"]]
    assert kinds == [
        ("GW-DATA-009", "score", "interrupted"),
        ("GW-DATA-009", "score", "done"),
        ("GW-DATA-001", "sanity", "done"),
        ("GW-DATA-003", "train", "failed"),
        ("GW-DATA-003", "train", "done"),
    ]
    assert res["experiments"]["GW-DATA-003"]["phase"] == "11"
    assert res["experiments"]["GW-DATA-009"]["phase"] == "11"
