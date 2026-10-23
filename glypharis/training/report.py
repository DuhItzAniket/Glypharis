"""Run report for a recognizer training run (EXPERIMENT_PROTOCOL §8).

Built only from what the run left on disk, so it can be regenerated at any time:
``<run_dir>/metrics.jsonl``, ``summary.json`` and ``val_best_samples.jsonl``. Writes
``report.md``, ``curves.png`` and ``worst_lines.png`` into the report directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from glypharis.eval.stats import ratio_ci
from glypharis.eval.text_metrics import score_pair

WORST = 20


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _curves(steps: list[dict[str, Any]], epochs: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes.ravel()
    if steps:
        x = [s["step"] for s in steps]
        ax[0].plot(x, [s["loss"] for s in steps], lw=0.8, label="train (running)")
    if epochs:
        ex = [e["step"] for e in epochs]
        ax[0].plot(ex, [e["val_loss"] for e in epochs], "o-", label="val")
        ax[0].plot(ex, [e["train_eval_loss"] for e in epochs], "s--", label="train sample")
        ax[1].plot(ex, [e["val_cer"] for e in epochs], "o-", label="val CER")
        ax[1].plot(ex, [e["train_cer"] for e in epochs], "s--", label="train CER")
        ax[1].plot(ex, [e["val_wer"] for e in epochs], "^-", label="val WER")
        ax[5].plot(ex, [e["blank_ratio"] for e in epochs], "o-", label="blank share")
        ax[5].plot(ex, [e["empty_ratio"] for e in epochs], "s-", label="empty decodes")
        if "vram_peak_mib" in epochs[0]:
            ax[4].plot(ex, [e["vram_peak_mib"] for e in epochs], "o-", label="peak VRAM MiB")
    if steps:
        ax[2].plot(x, [s["lr"] for s in steps])
        ax[3].plot(x, [s["grad_norm"] for s in steps], lw=0.6)
        ax[3].set_yscale("log")
        if "vram_peak_mib" in steps[0]:
            ax[4].plot(x, [s["vram_peak_mib"] for s in steps], lw=0.6, label="running peak")
    ax[0].set_yscale("log")  # the first steps' loss is an order of magnitude above the rest
    titles = ["loss (log scale)", "CER / WER", "learning rate", "gradient norm (before clipping)",
              "VRAM", "collapse checks"]  # fmt: skip
    for a, t in zip(ax, titles, strict=True):
        a.set_title(t)
        a.set_xlabel("optimizer step")
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _worst_strip(worst: list[dict[str, Any]], images: dict[str, Any], out: Path) -> bool:
    from PIL import Image, ImageDraw

    tiles = []
    for w in worst:
        img = images.get(w["key"])
        if img is None:
            continue
        img = img.convert("L")
        scale = 32 / img.height
        img = img.resize((max(1, min(900, round(img.width * scale))), 32))
        tile = Image.new("L", (900, 58), 255)
        tile.paste(img, (0, 0))
        ImageDraw.Draw(tile).text((2, 36), f"pred: {w['pred'][:110]}", fill=0)
        tiles.append(tile)
    if not tiles:
        return False
    sheet = Image.new("L", (900, 60 * len(tiles)), 255)
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (0, 60 * i))
    sheet.save(out)
    return True


def write_report(run_dir: Path, report_dir: Path, images: dict[str, Any] | None = None) -> Path:
    """Write report.md (+ figures). ``images`` maps line keys to crops for the worst lines."""
    report_dir.mkdir(parents=True, exist_ok=True)
    recs = _read_jsonl(run_dir / "metrics.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    steps = [r for r in recs if r["type"] == "step"]
    epochs = [r for r in recs if r["type"] == "epoch"]
    samples = _read_jsonl(run_dir / "val_best_samples.jsonl")
    _curves(steps, epochs, report_dir / "curves.png")

    errs, chars = [], []
    for s in samples:
        sc = score_pair(s["pred"], s["label"])
        errs.append(sc.char_errors)
        chars.append(sc.ref_chars)
    ci = ratio_ci(np.asarray(errs, float), np.asarray(chars, float), "lines") if samples else None
    worst = sorted(samples, key=lambda s: (-s["cer"], s["key"]))[:WORST]
    strip = _worst_strip(worst, images or {}, report_dir / "worst_lines.png")

    best, final = summary.get("best") or {}, summary.get("final") or {}
    hours = summary["train_seconds"] / 3600

    def row(name: str, p: dict[str, Any]) -> str:
        if not p:
            return f"| {name} | - | - | - | - | - | - |"
        cells = [p["epoch"], p["step"], f"{100 * p['val_cer']:.2f}", f"{100 * p['val_wer']:.2f}",
                 f"{p['val_loss']:.4f}", f"{100 * p['train_cer']:.2f}"]  # fmt: skip
        return f"| {name} | " + " | ".join(str(c) for c in cells) + " |"

    git = summary.get("git") or {}
    device = summary.get("gpu") or "CPU"
    time_note = f"{hours:.2f} GPU-hours" if summary.get("gpu") else "CPU"
    data = "; ".join(f"`{k}` {v['count']:,} lines, sha256 {v['bin_sha256'][:12]}"
                     for k, v in summary["data"].items())  # fmt: skip
    lines = [
        f"# Run report: {summary['experiment_id']}",
        "",
        "Generated by `glypharis.training.report`; do not edit by hand.",
        "",
        "## Run",
        "",
        f"- Config hash `{summary['config_hash'][:16]}`, git `{git.get('commit')}` "
        f"(dirty: {git.get('dirty')})",
        f"- Device {summary['device']} ({device}), precision {summary['precision']}, "
        f"torch {summary['torch']}",
        f"- Parameters {summary['parameters']:,}; {summary['train_lines']:,} training lines, "
        f"{summary['val_lines']:,} validation lines",
        f"- Epochs run {summary['epochs_run']}, optimizer steps {summary['steps']:,}, stopped "
        f"early: {summary['stopped_early']}; training time {hours:.2f} h ({time_note})",
        f"- Data: {data}",
        "",
        "## Best and final epoch",
        "",
        "| | Epoch | Step | Val CER % | Val WER % | Val loss | Train CER % |",
        "|---|---|---|---|---|---|---|",
        row("best (selected)", best),
        row("final", final),
        "",
        "The selected checkpoint is `best.pt`: the lowest validation CER (strict track), as the "
        "protocol requires (§7); the final epoch is shown for comparison.",
        "",
    ]  # fmt: skip
    if ci is not None:
        lines += [f"Validation CER of the selected checkpoint: {100 * ci.value:.2f}% "
                  f"[{100 * ci.low:.2f}, {100 * ci.high:.2f}] (95% bootstrap CI over "
                  f"{len(samples):,} lines).", ""]  # fmt: skip
    lines += ["## Diagnostics", ""]
    flags = summary.get("flags") or []
    if flags:
        lines += ["| Flag | Epoch | Step | Evidence |", "|---|---|---|---|"]
        lines += [f"| {f['name']} | {f['epoch']} | {f['step']} | {f['evidence']} |" for f in flags]
    else:
        lines.append("No flags raised.")
    lines += ["", "## Curves", "", "![curves](curves.png)", "", "## Worst validation lines", ""]
    lines += ["| Key | Reference | Prediction | CER |", "|---|---|---|---|"]
    for w in worst:
        ref = w["label"].replace("|", "\\|")
        pred = w["pred"].replace("|", "\\|")
        lines.append(f"| {w['key']} | {ref} | {pred} | {w['cer']:.2f} |")
    if strip:
        lines += ["", "![worst lines](worst_lines.png)"]
    path = report_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path
