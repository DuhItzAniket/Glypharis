from __future__ import annotations

import copy
import csv
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import experiment_comparison as ec
import pytest

REPO = Path(__file__).resolve().parents[1]


def metric(value: float, unit: str) -> dict[str, Any]:
    return {"value": value, "low": value - 0.01, "high": value + 0.01, "unit": unit}


def rec_record(experiment: str, system: str, eval_set: str, cer: float) -> dict[str, Any]:
    unit = "words" if eval_set == "iiit5k_test" else "lines"
    strict = {"samples": 100, "cer": metric(cer, unit), "wer": metric(cer * 2, unit)}
    tracks: dict[str, Any] = {"strict": strict, "normalised": copy.deepcopy(strict)}
    if eval_set == "iiit5k_test":
        tracks["alnum"] = {"samples": 100, "cer": metric(cer / 2, unit), "wer": metric(cer, unit)}
    return {
        "experiment": experiment,
        "system": system,
        "task": "rec",
        "eval_set": eval_set,
        "tracks": tracks,
    }


def e2e_record(experiment: str, system: str, eval_set: str, cer: float) -> dict[str, Any]:
    strict: dict[str, Any] = {
        "pages": 10,
        "page_cer_bag_of_lines": metric(cer, "pages"),
        "e2e_f1": metric(0.5, "pages"),
    }
    detection: dict[str, Any] | None = {"iou_0.5": {"f1": metric(0.7, "pages")}}
    if eval_set == "gdc_test":
        strict["page_cer_order_aware"] = metric(cer / 2, "pages")
        strict["page_wer_order_aware"] = metric(cer, "pages")
    if eval_set == "omnidocbench_test":
        strict = {
            "pages": 10,
            "page_cer_bag_of_lines": None,
            "e2e_f1": None,
            "page_cer_order_aware": metric(cer, "pages"),
            "page_wer_order_aware": metric(cer * 2, "pages"),
        }
        detection = None
    return {
        "experiment": experiment,
        "system": system,
        "task": "e2e",
        "eval_set": eval_set,
        "tracks": {"strict": strict, "normalised": copy.deepcopy(strict)},
        "detection": detection,
    }


def bench(system: str, experiment: str, task: str, ms: float, mib: float, threads: Any, sit: str):
    return {
        "system": system,
        "experiment": experiment,
        "task": task,
        "mean_ms": ms,
        "p95_ms": ms * 2,
        "peak_mem_mib": 100.0,
        "model_mib": mib,
        "threads": threads,
        "sitting": sit,
    }


def profile(experiment: str, task: str, ms: float, mib: float, threads: Any) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "task": task,
        "eval_set": "hiertext_val",
        "batch": 1,
        "mean_ms": ms,
        "p95_ms": ms * 2,
        "peak_mem_mib": 100.0,
        "model_mib": mib,
        "threads": threads,
    }


def cost(label: str, task: str, runtime: str, ms: float, batch: int = 1, vram=None):
    row: dict[str, Any] = {
        "label": label,
        "task": task,
        "runtime": runtime,
        "batch": batch,
        "mean_ms": ms,
        "peak_vram": vram,
    }
    if task == "rec":
        row["mean_line_ms"] = ms / batch
    return row


def sources() -> dict[str, dict[str, Any]]:
    threads = {"cpu_threads": 10}
    benchmark_rows = [
        bench("Base A", "GW-BASE-001", "rec", 50.0, 5.0, threads, "Phase 5"),
        bench("Base A", "GW-BASE-001", "e2e", 1500.25, 9.0, threads, "Phase 5"),
        bench("Base B", "GW-BASE-002", "rec", 20.0, 2.0, {"OMP_THREAD_LIMIT": None}, "Phase 5"),
        bench("GLYPHARIS v1 PyTorch, CPU, 10 threads", "GW-EFF-007", "rec", 30.0, 30.0, 10,
              "Phase 16"),
        bench("GLYPHARIS v1 PyTorch, CPU, 10 threads", "GW-EFF-007", "e2e", 2500.0, 42.0, 10,
              "Phase 16"),
    ]  # fmt: skip
    return {
        ec.EXTERNAL: {
            "predictions": [
                rec_record("GW-REC-039", "glypharis", "gdc_test", 0.04),
                rec_record("GW-REC-039", "glypharis", "iiit5k_test", 0.4),
                rec_record("GW-BASE-001", "base_a", "gdc_test", 0.06),
                rec_record("GW-BASE-001", "base_a", "iiit5k_test", 0.5),
                e2e_record("GW-E2E-007", "glypharis", "gdc_test", 0.2),
                e2e_record("GW-E2E-007", "glypharis", "omnidocbench_test", 0.3),
                e2e_record("GW-BASE-001", "base_a", "gdc_test", 0.4),
                e2e_record("GW-BASE-002", "base_b", "gdc_test", 0.45),
                {**e2e_record("GW-E2E-008", "glypharis", "gdc_test", 0.1), "task": "oracle"},
            ],
            "not_run": {"base_a e2e omnidocbench_test": "not run: too slow"},
        },
        ec.BASELINES: {
            "predictions": [
                rec_record("GW-BASE-001", "base_a", "gdc_test", 0.06),
                rec_record("GW-BASE-001", "base_a", "hiertext_val", 0.07),
            ],
            "efficiency": [
                profile("GW-BASE-001", "rec", 50.0, 5.0, threads),
                profile("GW-BASE-001", "e2e", 1500.25, 9.0, threads),
                profile("GW-BASE-002", "rec", 20.0, 2.0, {"OMP_THREAD_LIMIT": None}),
            ],
        },
        ec.PHASE16: {
            "benchmark": {"rows": copy.deepcopy(benchmark_rows)},
            "gw_eff_006": {
                "files": {
                    "detector.pt": {"mib": 12.0},
                    "detector.onnx": {"mib": 11.0},
                    "recognizer.pt": {"mib": 30.0},
                    "recognizer.onnx": {"mib": 31.0},
                }
            },
            "cost": {
                "parameters": {"detector": 300, "recognizer": 700, "total": 1000},
                "rows": {
                    "a": cost("PyTorch, CPU, 10 threads", "rec", "torch (CPU)", 30.0),
                    "b": cost("PyTorch, GPU, batch 32", "rec", "torch (GPU)", 320.0, 32,
                              {"max_memory_allocated_mib": 50.5}),
                    "c": cost("PyTorch, CPU, 10 threads", "e2e", "torch (cpu)", 2500.0),
                    "d": cost("fp32 ONNX, CPU, 10 threads", "e2e", "onnxruntime 1 (CPU)", 8000.0),
                },
            },
        },
        ec.BENCHMARK: {"rows": benchmark_rows, "eval_set": "hiertext_val"},
    }  # fmt: skip


def write_sources(root: Path, data: dict[str, dict[str, Any]]) -> None:
    for path, content in data.items():
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(json.dumps(content), encoding="utf-8", newline="\n")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write_sources(tmp_path, sources())
    return tmp_path


def run(root: Path, *args: str) -> int:
    return ec.main(["--root", str(root), *args])


def csv_rows(root: Path) -> list[dict[str, str]]:
    text = (root / ec.OUT_CSV).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def find(rows: list[dict[str, str]], **wanted: str) -> dict[str, str]:
    found = [r for r in rows if all(r[k] == v for k, v in wanted.items())]
    assert len(found) == 1, wanted
    return found[0]


@pytest.fixture
def rows(repo: Path) -> list[dict[str, str]]:
    assert run(repo) == 0
    return csv_rows(repo)


def test_every_row_names_its_accuracy_and_latency_experiments(rows):
    row = find(rows, model="GLYPHARIS v1", table="recognition", dataset="GDC test")
    assert (row["accuracy_id"], row["latency_id"]) == ("GW-REC-039", "GW-EFF-007")
    row = find(rows, model="Base A", table="end_to_end", dataset="GDC test")
    assert (row["accuracy_id"], row["latency_id"]) == ("GW-BASE-001", "GW-BASE-001")
    assert all(r["latency_id"] != "n/a" for r in rows if r["table"] == "runtime")


def test_fractions_become_percentages_with_their_interval(rows):
    row = find(rows, model="GLYPHARIS v1", table="recognition", dataset="GDC test")
    assert (row["cer_pct"], row["cer_low"], row["cer_high"]) == ("4.00", "3.00", "5.00")
    assert row["wer_pct"] == "8.00"
    assert row["interval_unit"] == "lines"


def test_cost_columns_come_from_the_named_rows(rows):
    row = find(rows, model="GLYPHARIS v1", table="end_to_end", dataset="GDC test")
    assert row["latency_ms"] == "2,500.0"
    assert row["latency_unit"] == "ms per page"
    assert row["latency_setup"] == "PyTorch, CPU, 10 threads"
    assert row["params"] == "1,000"
    assert row["model_mib"] == "42.00"
    row = find(rows, model="GLYPHARIS v1", table="recognition", dataset="GDC test")
    assert row["params"] == "700"
    row = find(rows, model="Base A", table="end_to_end", dataset="GDC test")
    assert (row["latency_ms"], row["latency_sitting"]) == ("1,500.2", "Phase 5")
    assert row["latency_setup"] == "CPU, 10 threads"


def test_missing_values_are_na_and_never_filled_in(rows):
    base = find(rows, model="Base A", table="recognition", dataset="GDC test")
    assert base["params"] == "n/a"  # no source gives a baseline's parameters
    assert base["vram_mib"] == "n/a"
    assert base["f1_pct"] == "n/a"  # recognition has no F1
    # Base B has a recognition profile but none for pages: its page row stays without one.
    page = find(rows, model="Base B", table="end_to_end", dataset="GDC test")
    assert page["latency_ms"] == page["model_mib"] == page["latency_id"] == "n/a"
    assert page["cer_pct"] == "45.00"


def test_order_aware_metrics_only_where_the_source_has_them(rows):
    row = find(rows, model="GLYPHARIS v1", table="end_to_end", dataset="GDC test")
    assert (row["cer_order_aware_pct"], row["wer_pct"]) == ("10.00", "20.00")
    assert (row["f1_pct"], row["detection_f1_pct"]) == ("50.00", "70.00")
    row = find(rows, model="GLYPHARIS v1", table="end_to_end", dataset="OmniDocBench test")
    assert row["cer_pct"] == row["f1_pct"] == row["detection_f1_pct"] == "n/a"
    assert row["cer_order_aware_pct"] == "30.00"


def test_iiit5k_shows_strict_and_the_alphanumeric_headline(rows):
    strict = find(rows, model="GLYPHARIS v1", dataset="IIIT5K test", track="strict")
    alnum = find(rows, model="GLYPHARIS v1", dataset="IIIT5K test",
                 track="alphanumeric (headline)")  # fmt: skip
    assert (strict["cer_pct"], alnum["cer_pct"]) == ("40.00", "20.00")
    assert strict["interval_unit"] == "words"


def test_only_strict_and_headline_tracks_are_shown(rows):
    assert {r["track"] for r in rows} == {"strict", "alphanumeric (headline)", "n/a"}


def test_oracle_runs_are_left_out(rows):
    assert not [r for r in rows if r["accuracy_id"] == "GW-E2E-008"]


def test_baseline_validation_results_come_from_phase_5(rows):
    row = find(rows, model="Base A", dataset="HierText validation", table="recognition")
    assert row["cer_pct"] == "7.00"


def test_runtime_rows_hold_vram_and_per_line_batch_latency(rows):
    gpu = find(rows, table="runtime", latency_setup="PyTorch, GPU, batch 32")
    assert gpu["vram_mib"] == "50.5"
    assert gpu["latency_ms"] == "10.0"  # 320 ms per batch of 32, not the batch time
    assert gpu["accuracy_id"] == "n/a"
    cpu = find(rows, table="runtime", task="rec", latency_setup="PyTorch, CPU, 10 threads")
    assert cpu["vram_mib"] == "n/a"
    onnx = find(rows, table="runtime", latency_setup="fp32 ONNX, CPU, 10 threads")
    assert onnx["model_mib"] == "42.00"  # the ONNX files, 11 + 31


def test_header_note_forbids_latency_differences(repo):
    assert run(repo) == 0
    text = (repo / ec.OUT_MD).read_text(encoding="utf-8")
    assert "Latency is measured on HierText validation, in different sittings" in text
    assert "no latency difference may be stated" in text
    assert "Base A, e2e, OmniDocBench test: not run: too slow." in text


def test_markdown_shows_na_and_intervals(repo):
    assert run(repo) == 0
    text = (repo / ec.OUT_MD).read_text(encoding="utf-8")
    assert "| 4.00 [3.00, 5.00] | 8.00 [7.00, 9.00] | n/a | 30.0 |" in text


def set_value(path: str, *steps: Any) -> Callable[[dict[str, Any]], None]:
    """A change to one source: the keys to walk, then the last key, then its new value."""
    *keys, last, value = steps

    def change(data: dict[str, Any]) -> None:
        node = data[path]
        for key in keys:
            node = node[key]
        node[last] = value

    return change


DISAGREE = [
    ("differ on", set_value(ec.BASELINES, "predictions", 0, "tracks", "strict", "samples", 99)),
    ("different benchmark rows", set_value(ec.PHASE16, "benchmark", "rows", 0, "mean_ms", 51.0)),
    ("differs from", set_value(ec.BASELINES, "efficiency", 0, "mean_ms", 51.0)),
    ("not hiertext_val at batch 1", set_value(ec.BASELINES, "efficiency", 0, "batch", 8)),
    ("has no row in", set_value(ec.BASELINES, "efficiency", 0, "experiment", "GW-BASE-007")),
    ("latency is on", set_value(ec.BENCHMARK, "eval_set", "gdc_test")),
    ("do not add up", set_value(ec.PHASE16, "gw_eff_006", "files", "recognizer.pt", "mib", 29.0)),
    ("mean_ms differs", set_value(ec.PHASE16, "cost", "rows", "c", "mean_ms", 1.0)),
    ("mean_line_ms is not", set_value(ec.PHASE16, "cost", "rows", "b", "mean_line_ms", 320.0)),
    ("unknown evaluation set", set_value(ec.EXTERNAL, "predictions", 0, "eval_set", "new_set")),
    ("has no name", set_value(ec.EXTERNAL, "predictions", 2, "experiment", "GW-BASE-005")),
    ("not_run key", set_value(ec.EXTERNAL, "not_run", {"odd key": "x"})),
]


@pytest.mark.parametrize(("message", "change"), DISAGREE)
def test_disagreeing_sources_fail_and_write_nothing(tmp_path, capsys, message, change):
    data = sources()
    change(data)
    write_sources(tmp_path, data)
    assert run(tmp_path) == 1
    err = capsys.readouterr().err
    assert message in err, err
    assert not (tmp_path / ec.OUT_MD).exists()
    assert not (tmp_path / ec.OUT_CSV).exists()


def test_a_missing_field_fails_cleanly(tmp_path, capsys):
    data = sources()
    del data[ec.PHASE16]["cost"]["parameters"]
    write_sources(tmp_path, data)
    assert run(tmp_path) == 1
    assert "unusable source" in capsys.readouterr().err
    assert not (tmp_path / ec.OUT_CSV).exists()


def test_a_missing_source_fails_cleanly(repo, capsys):
    (repo / ec.BENCHMARK).unlink()
    assert run(repo) == 1
    assert "cannot read" in capsys.readouterr().err


def test_personal_paths_are_refused(tmp_path, capsys):
    data = sources()
    data[ec.EXTERNAL]["not_run"] = {"base_a e2e omnidocbench_test": "see C:\\Users\\someone"}
    write_sources(tmp_path, data)
    assert run(tmp_path) == 1
    assert "R11" in capsys.readouterr().err
    assert not (tmp_path / ec.OUT_MD).exists()


def test_output_is_deterministic_with_lf_only(repo):
    assert run(repo) == 0
    first = {p: (repo / p).read_bytes() for p in (ec.OUT_MD, ec.OUT_CSV)}
    assert run(repo) == 0
    assert first == {p: (repo / p).read_bytes() for p in (ec.OUT_MD, ec.OUT_CSV)}
    for path, data in first.items():
        assert b"\r" not in data, path
        assert str(repo).encode() not in data, path


def test_check_current_stale_and_crlf(repo, capsys):
    assert run(repo, "--check") == 1  # nothing generated yet
    assert not (repo / ec.OUT_MD).exists()
    assert run(repo) == 0
    assert run(repo, "--check") == 0
    full = repo / ec.OUT_CSV
    full.write_bytes(full.read_bytes().replace(b"\n", b"\r\n"))
    assert run(repo, "--check") == 0
    full.write_text("old\n", encoding="utf-8", newline="\n")
    capsys.readouterr()
    assert run(repo, "--check") == 1
    assert f"{ec.OUT_CSV} is out of date" in capsys.readouterr().err
    assert full.read_text(encoding="utf-8") == "old\n"


def test_committed_table_is_current():
    assert ec.main(["--root", str(REPO), "--check"]) == 0
