from __future__ import annotations

import json

import compute_ledger
import pytest


@pytest.fixture(autouse=True)
def _no_default_logs(tmp_path, monkeypatch):
    # The default log paths are relative: run from an empty folder so no real log is read.
    monkeypatch.chdir(tmp_path)


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


def test_phase13_progress_log_keeps_only_its_lines_and_hand_stops(tmp_path):
    gpu = tmp_path / "phase_progress.log"
    gpu.write_text(
        "2026-10-28 00:27:06 START P10 confidence calibration (temperature scaling, ECE)\n"
        "2026-10-30 15:34:06 START P13 GLYPH-C score GW-REC-006\n"
        "2026-10-30 15:44:00 DONE  P13 GLYPH-C score GW-REC-006 :: mCE 0.0687\n"
        "2026-10-31 10:00:00 START P13 train GW-REC-028\n"
        "2026-10-31 13:00:00 STOPPED before P13 train GW-REC-028 (owner asked to pause)\n"
        "2026-10-31 14:00:00 STOPPED before P13 train GW-REC-029\n"
        "2026-10-31 15:00:00 START P13 train GW-REC-028\n"
        "2026-10-31 17:00:00 DONE  P13 train GW-REC-028 :: done\n"
        "2026-10-31 17:00:00 START P13 report GW-REC-028\n"
        "2026-10-31 17:00:10 DONE  P13 report GW-REC-028 :: report\n"
        "2026-11-04 05:13:53 START P13 det eval GW-DET-001 (Phase 9 settings, all pages)\n"
        "2026-11-04 05:19:19 DONE  P13 det eval GW-DET-001 (Phase 9 settings, all pages) :: }\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert compute_ledger.main(["--gpu-logs", str(gpu), "--cpu-logs", "--runs", str(tmp_path),
                                "--out", str(out)]) == 0  # fmt: skip
    res = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    kinds = [(j["experiment"], j["kind"], j["outcome"], j["hours"]) for j in res["jobs"]]
    assert kinds[0][:3] == ("GW-REC-037", "score", "done")
    assert abs(kinds[0][3] - 594 / 3600) < 1e-9
    assert kinds[1:3] == [
        ("GW-REC-028", "train", "stopped", 3.0),
        ("GW-REC-028", "train", "done", 2.0),
    ]
    assert kinds[3][:3] == ("GW-DET-007", "score", "done")
    assert len(kinds) == 4  # the P10 line, the step never begun and the report are not jobs
    assert res["experiments"]["GW-REC-028"]["failed_jobs"] == 1
    assert res["experiments"]["GW-REC-028"]["phase"] == "13"
    assert res["experiments"]["GW-DET-007"]["phase"] == "13"


PHASE16_LOG = (
    "2026-11-17T13:44:59Z CHAIN START (6 steps)\n"
    "2026-11-17T13:44:59Z START 1/6 eff006_ref_write\n"
    "2026-11-17T14:44:59Z DONE 1/6 eff006_ref_write (3600s)\n"
    "2026-11-17T14:45:00Z START 2/6 eff007_l_torch\n"
    "2026-11-17T14:51:00Z DONE 2/6 eff007_l_torch (360s)\n"
    "2026-11-17T14:51:00Z START 3/6 eff007_p_cuda\n"
    "2026-11-17T15:09:00Z FAIL 3/6 eff007_p_cuda (exit 1, 1080s; see the step log)\n"
    "2026-11-17T15:10:00Z RESUME after FAIL 3 (a fix)\n"
    "2026-11-17T15:10:00Z START 3/6 eff007_p_cuda\n"
    "2026-11-17T15:46:00Z DONE 3/6 eff007_p_cuda (2160s)\n"
    "2026-11-17T23:30:00Z START 4/6 eff010_baseline\n"
    "2026-11-18T00:15:00Z DONE 4/6 eff010_baseline (2700s)\n"
    "2026-11-18T00:15:00Z START 5/6 eff009_eval_lines_rec-GW-REC-020\n"
    "2026-11-18T00:27:00Z DONE 5/6 eff009_eval_lines_rec-GW-REC-020 (720s)\n"
    "2026-11-18T00:27:00Z START 6/6 score\n"
    "2026-11-18T00:30:00Z DONE 6/6 score (180s)\n"
    "2026-11-18T00:30:00Z ALL DONE\n"
)


def test_phase16_chain_lines_pair_by_step(tmp_path):
    log = tmp_path / "phase16_chain.log"
    log.write_text(PHASE16_LOG, encoding="utf-8")
    jobs = compute_ledger.parse_phase16(log)
    got = [(j.device, j.experiment, j.kind, j.detail, j.outcome, round(j.hours * 60))
           for j in jobs]  # fmt: skip
    assert got == [  # CHAIN START, RESUME and ALL DONE are not jobs
        ("cpu", "GW-EFF-006", "efficiency", "eff006_ref_write", "done", 60),
        ("cpu", "GW-EFF-007", "efficiency", "eff007_l_torch", "done", 6),
        ("gpu", "GW-EFF-007", "efficiency", "eff007_p_cuda", "failed (exit 1)", 18),
        ("gpu", "GW-EFF-007", "efficiency", "eff007_p_cuda", "done", 36),
        ("cpu", "GW-BASE-002", "baseline", "eff010_baseline", "done", 45),
        ("cpu", "GW-EFF-009", "efficiency", "eff009_eval_lines_rec-GW-REC-020", "done", 12),
        ("cpu", "Phase 16 scoring", "score", "score", "done", 3),
    ]
    assert jobs[0].start == "2026-11-17T13:44:59Z"  # the UTC stamp as logged
    assert jobs[4].hours == 0.75  # across midnight


def test_phase16_end_lines_without_their_start_and_open_steps(tmp_path):
    log = tmp_path / "phase16_chain.log"
    log.write_text(
        "2026-11-17T13:00:00Z DONE 1/3 eff006_bundle (12s)\n"
        "2026-11-17T13:00:00Z START 2/3 eff008_eval_v1-onnx\n"
        "2026-11-17T13:30:00Z START 2/3 eff008_eval_v1-onnx\n"
        "2026-11-17T13:40:00Z DONE 2/3 eff008_eval_v1-onnx (600s)\n"
        "2026-11-17T13:40:00Z START 3/3 eff011_time_rec-GW-REC-003_1\n",
        encoding="utf-8",
    )
    got = [(j.experiment, j.outcome, j.hours) for j in compute_ledger.parse_phase16(log)]
    assert got == [
        ("GW-EFF-008", "interrupted", None),
        ("GW-EFF-008", "done", 1 / 6),
        ("GW-EFF-011", "running", None),
    ]


def test_phase16_rows_and_totals(tmp_path):
    log = tmp_path / "phase16_chain.log"
    log.write_text(PHASE16_LOG, encoding="utf-8")
    cpu = tmp_path / "chain.log"
    cpu.write_text(
        "2026-10-21 21:05:54 START GW-BASE-002 ppocr_mobile rec glyphsynth_val\n"
        "2026-10-21 21:35:54 DONE  GW-BASE-002 ppocr_mobile rec glyphsynth_val :: 10000 new\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert compute_ledger.main(["--gpu-logs", "--cpu-logs", str(cpu), "--phase16-logs",
                                str(log), "--runs", "--out", str(out)]) == 0  # fmt: skip
    res = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    exps = {k: (v["phase"], v["device"], v["jobs"], v["failed_jobs"], round(v["wall_h"] * 60))
            for k, v in res["experiments"].items()}  # fmt: skip
    assert exps == {
        "GW-BASE-002": ("05", "cpu", 1, 0, 30),  # Phase 5's job keeps its own row
        "GW-BASE-002 (Phase 16 amendment)": ("16", "cpu", 1, 0, 45),
        "GW-EFF-006": ("16", "cpu", 1, 0, 60),
        "GW-EFF-007": ("16", "cpu", 1, 0, 6),
        "GW-EFF-007 (GPU jobs)": ("16", "gpu", 2, 1, 54),
        "GW-EFF-009": ("16", "cpu", 1, 0, 12),
        "Phase 16 scoring": ("16", "cpu", 1, 0, 3),
    }
    totals = res["totals"]
    assert abs(totals["gpu_h"] - 0.9) < 1e-9 and abs(totals["gpu_failed_h"] - 0.3) < 1e-9
    assert abs(totals["cpu_h"] - (0.5 + 126 / 60)) < 1e-9 and totals["cpu_failed_h"] == 0
    md = (out / "ledger.md").read_text(encoding="utf-8")
    assert "| 05 | 0.0 | 0.0 | 0.5 | 0.0 |" in md
    assert "| 16 | 0.9 | 0.3 | 2.1 | 0.0 |" in md
    assert "| GW-EFF-007 (GPU jobs) | 16 | GPU | 2 | 1 | 0.90 |  |" in md


def test_phase_of_phase16_rows():
    assert compute_ledger.phase_of("GW-EFF-006") == "16"
    assert compute_ledger.phase_of("GW-EFF-011") == "16"
    assert compute_ledger.phase_of("GW-BASE-002") == "05"
    assert compute_ledger.phase_of("GW-BASE-002 (Phase 16 amendment)") == "16"
    assert compute_ledger.phase_of("Phase 16 scoring") == "16"
