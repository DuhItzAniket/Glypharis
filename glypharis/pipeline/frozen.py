"""The frozen GLYPHARIS v1 model, built from its freeze record by code (Phase 14).

    frozen = load_frozen("cuda")          # reads configs/final/glypharis_v1.yaml
    frozen.ocr.predict(page)              # detector + recognizer at the recorded settings
    frozen.provenance                     # hashes, versions, numerics flags, git state

Every setting comes from the record (`FREEZE_RECORD`), never from a command line: the detector's
resolution, post-processing and operating point (Phase 9), and the recognizer's input height,
maximum width and temperature. Before anything is built, both checkpoint files are hashed and
compared with the record, and the character list stored in the recognizer checkpoint is compared
with the charset file; any difference is refused. After building, the settings the objects
actually hold are compared with the record once more, so a default that slipped in is caught.

Numerics (registered in docs/PHASE_STATUS.md, Phase 14): the recognizer runs in strict fp32 (no
TF32), as the detector already does; cuDNN autotuning is off and deterministic algorithms are on
(`strict_inference`).

`pinned_code_check` ties a run to the ``phase14-eval`` tag: nothing tracked or untracked may
differ from the tag except the records a run writes (`RUN_RECORDS`: the access log, the outputs
manifest, research/failures.md, where a void sitting is recorded, and reports/). Rename detection
is off, so a pinned file moved into a record path still counts as changed. The code trees
(`PINNED_TREES`) are named in the error message.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml

from glypharis.config.loader import from_dict
from glypharis.config.schema import DetInferConfig, DetPostConfig, ModelConfig
from glypharis.data.download import sha256_file
from glypharis.data.text import Charset
from glypharis.models.det.detector import TextLineDetector
from glypharis.models.rec.crnn import build_recognizer
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer
from glypharis.training.checkpoint import load_checkpoint
from glypharis.utils.gitinfo import git_info

FREEZE_RECORD = Path("configs/final/glypharis_v1.yaml")
PIN_TAG = "phase14-eval"
PINNED_TREES = ("glypharis", "scripts", "configs", "requirements")
# The only paths that may change after the tag (Phase 14 plan, item 1, as amended: a void
# latency sitting is recorded in research/failures.md): a file, or a folder when the entry ends
# in "/".
RUN_RECORDS = (
    "research/test_access_log.md",
    "research/failures.md",
    "data/manifests/processed/phase14_outputs.json",
    "reports/",
)
# data/manifests/charset.json as recorded with the Phase 14 plan.
CHARSET_SHA256 = "bbdfab42f62b6a08df9ceffc1119eb304c1f9ad741285035737ac958803e803c"
_CUBLAS_WORKSPACE = ":4096:8"  # deterministic cuBLAS; as glypharis.utils.seed


class FrozenModelError(RuntimeError):
    """The files or settings differ from the freeze record; nothing is run."""


@dataclass
class Frozen:
    ocr: GlypharisOCR  # detector + recognizer, Phase 9 settings from the record
    recognizer: TextRecognizer  # the same object as ocr.recognizer
    provenance: dict[str, Any]

    def settings(self) -> dict[str, Any]:
        """The settings the built objects hold, in the record's terms (what a run records)."""
        return _settings(self.ocr)


def strict_inference(device: str) -> dict[str, Any]:
    """Switch PyTorch to reproducible fp32 inference and return the flags that were set.

    cuDNN autotuning off and deterministic kernels on (an op without one only warns, as in
    training); TF32 off for cuDNN and CUDA matmuls; float32 matmul precision "highest". The
    recognizer and detector also enter `glypharis.models.det.runtime.strict_fp32` around every
    forward pass, which states the same for the block.
    """
    if torch.device(device).type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "recognizer_strict_fp32": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _repo_path(root: Path, value: str, what: str) -> Path:
    """A record path, which must be repo-relative, resolved under ``root``."""
    p = PurePosixPath(value)
    if p.is_absolute() or Path(value).is_absolute() or ".." in p.parts:
        raise FrozenModelError(f"{what} {value!r} in the record is not a repo-relative path")
    return root / Path(*p.parts)


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix() if not path.is_absolute() else path.name


def _verified(root: Path, section: dict[str, Any], what: str) -> tuple[Path, str]:
    path = _repo_path(root, section["checkpoint"], f"{what} checkpoint")
    if not path.is_file():
        raise FrozenModelError(f"{what} checkpoint {section['checkpoint']} is missing")
    sha = sha256_file(path)
    if sha != section["checkpoint_sha256"]:
        raise FrozenModelError(
            f"{what} checkpoint {section['checkpoint']} has SHA-256 {sha}, the record says "
            f"{section['checkpoint_sha256']}; refusing to run"
        )
    return path, sha


def _same(what: str, got: Any, want: Any) -> None:
    if got != want:
        raise FrozenModelError(f"{what}: {got!r} differs from the record's {want!r}")


def _check_training_record(ck: dict[str, Any], section: dict[str, Any], what: str) -> None:
    """The checkpoint is the run the record describes (experiment, config hash, commit)."""
    config = ck.get("config") or {}
    _same(f"{what} checkpoint experiment", config.get("experiment_id"), section["experiment"])
    _same(f"{what} checkpoint config hash", ck.get("config_hash"), section["config_hash"])
    _same(f"{what} checkpoint git commit", (ck.get("git") or {}).get("commit"),
          section["git_commit"])  # fmt: skip


def _recognizer(
    root: Path, rec: dict[str, Any], device: str, charset_sha256: str | None
) -> tuple[TextRecognizer, dict[str, Any]]:
    path, sha = _verified(root, rec, "recognizer")
    charset_path = _repo_path(root, rec["charset"], "charset")
    charset_sha = sha256_file(charset_path)
    if charset_sha256 is not None and charset_sha != charset_sha256:
        raise FrozenModelError(
            f"{rec['charset']} has SHA-256 {charset_sha}, the Phase 14 plan recorded "
            f"{charset_sha256}; refusing to run"
        )
    charset = Charset.load(charset_path)
    ck = load_checkpoint(path, "cpu")
    _check_training_record(ck, rec, "recognizer")
    if list(ck.get("charset") or []) != charset.characters:
        raise FrozenModelError(
            f"the character list stored in {rec['checkpoint']} differs from {rec['charset']}; "
            "refusing to run"
        )
    config = ck["config"]
    _same("recognizer model", config.get("model_name"), rec["model_name"])
    data = config.get("data") or {}
    height, max_width = int(rec["input"]["height"]), int(rec["input"]["max_width"])
    _same("recognizer input height in the checkpoint", data.get("image_height"), height)
    _same("recognizer maximum width in the checkpoint", data.get("max_width"), max_width)
    model_cfg = from_dict(ModelConfig, config["model"], "model")
    model = build_recognizer(charset.num_classes, model_cfg, rec["model_name"])
    model.load_state_dict(ck["model"])
    arch = {"model_name": rec["model_name"], "model": dataclasses.asdict(model_cfg)}
    recognizer = TextRecognizer(model, charset, height, max_width, device,
                                float(rec["temperature"]), arch, strict_fp32=True)  # fmt: skip
    files = {
        "recognizer_checkpoint": _rel(path, root),
        "recognizer_checkpoint_sha256": sha,
        "charset": _rel(charset_path, root),
        "charset_sha256": charset_sha,
    }
    return recognizer, files


def _detector(
    root: Path, det: dict[str, Any], device: str
) -> tuple[TextLineDetector, dict[str, Any]]:
    path, sha = _verified(root, det, "detector")
    ck = load_checkpoint(path, "cpu")
    _check_training_record(ck, det, "detector")
    _same("detector label version", ck.get("label_version"), det["label_version"])
    post = det["post"]
    names = {f.name for f in dataclasses.fields(DetPostConfig)}
    if set(post) != names:
        # A key left out would quietly take the schema default instead of the recorded value.
        raise FrozenModelError(
            f"the record's detector post-processing has keys {sorted(post)}, expected "
            f"{sorted(names)}"
        )
    del ck  # TextLineDetector reads the file again; no need to hold its weights twice
    base = TextLineDetector.from_checkpoint(path, device)
    spec = dataclasses.replace(
        base.spec,
        infer=from_dict(DetInferConfig, det["infer"], "detector.infer"),
        post=from_dict(DetPostConfig, post, "detector.post"),
        operating_point=str(det["operating_point"]),
    )
    files = {"detector_checkpoint": _rel(path, root), "detector_checkpoint_sha256": sha}
    return TextLineDetector(base.backend, spec), files


def _settings(ocr: GlypharisOCR) -> dict[str, Any]:
    spec = ocr.detector.spec
    rec = ocr.recognizer
    return {
        "detector": {
            "infer": dataclasses.asdict(spec.infer),
            "post": dataclasses.asdict(spec.post),
            "operating_point": spec.operating_point,
            "label_version": spec.label_version,
        },
        "recognizer": {
            "input": {"height": rec.height, "max_width": rec.max_width},
            "temperature": rec.temperature,
            "strict_fp32": rec.strict_fp32,
        },
        "crop": dataclasses.asdict(ocr.crop),
        "reading_order": ocr.order,
    }


def _check_settings(ocr: GlypharisOCR, record: dict[str, Any]) -> None:
    """What the built objects hold equals the record, value by value."""
    got = _settings(ocr)
    det, rec = record["detector"], record["recognizer"]
    _same("detector resolution", got["detector"]["infer"], det["infer"])
    _same("detector post-processing", got["detector"]["post"], det["post"])
    _same("detector operating point", got["detector"]["operating_point"], det["operating_point"])
    _same("detector label version", got["detector"]["label_version"], det["label_version"])
    _same("recognizer input", got["recognizer"]["input"], rec["input"])
    _same("recognizer temperature", got["recognizer"]["temperature"], rec["temperature"])
    _same("recognizer strict fp32", got["recognizer"]["strict_fp32"], True)


def _device_name(device: str) -> str:
    d = torch.device(device)
    if d.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(d)
    return d.type


def load_frozen(
    device: str,
    record: Path = FREEZE_RECORD,
    *,
    root: Path = Path("."),
    charset_sha256: str | None = CHARSET_SHA256,
) -> Frozen:
    """The frozen pipeline on ``device``, or `FrozenModelError` on any difference from the record.

    ``record`` is relative to ``root`` (the repository root) unless absolute; checkpoint and
    charset paths are the record's repo-relative paths under ``root``. ``charset_sha256`` is the
    charset file's hash fixed with the Phase 14 plan (``None`` skips that comparison; the
    character-list comparison with the checkpoint always runs).
    """
    root = Path(root)
    record_path = record if record.is_absolute() else root / record
    text = record_path.read_text(encoding="utf-8")
    rec_doc = yaml.safe_load(text)
    if not isinstance(rec_doc, dict) or "recognizer" not in rec_doc or "detector" not in rec_doc:
        raise FrozenModelError(f"{record.as_posix()} is not a freeze record")
    recognizer, rec_files = _recognizer(root, rec_doc["recognizer"], device, charset_sha256)
    detector, det_files = _detector(root, rec_doc["detector"], device)
    ocr = GlypharisOCR(detector, recognizer)
    ocr.model_version = str(rec_doc.get("name", ""))
    _check_settings(ocr, rec_doc)
    flags = strict_inference(device)  # only once everything matched: a refusal changes nothing
    git = git_info(root)
    provenance = {
        "record": _rel(record_path, root),
        "record_sha256": sha256_file(record_path),
        "name": rec_doc.get("name"),
        "recognizer_experiment": rec_doc["recognizer"]["experiment"],
        "detector_experiment": rec_doc["detector"]["experiment"],
        **rec_files,
        **det_files,
        "device": str(torch.device(device)),
        "device_name": _device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numerics": flags,
        "git_commit": git.commit,
        "git_dirty": git.dirty,
    }
    return Frozen(ocr, recognizer, provenance)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"git {' '.join(args)} could not run: {exc}") from exc


def _allowed(path: str) -> bool:
    return any(path == r or (r.endswith("/") and path.startswith(r)) for r in RUN_RECORDS)


def _status_paths(porcelain: str) -> list[str]:
    """Paths from ``git status --porcelain`` (both sides of a rename, should one be shown)."""
    out = []
    for ln in porcelain.splitlines():
        for part in ln[3:].split(" -> "):
            out.append(part.strip().strip('"'))
    return out


def tag_exists(tag: str = PIN_TAG, *, root: Path = Path(".")) -> bool:
    """Whether ``tag`` exists; a git error (not a repository, git missing) raises RuntimeError
    rather than reading as "no tag", so a step that must run before the tag fails closed."""
    found = _git(Path(root), "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    if found.returncode == 0 and found.stdout.strip():
        return True
    if found.returncode == 1:
        return False
    raise RuntimeError(f"git could not look up the tag {tag}: {found.stderr.strip()[:200]}")


def pinned_code_check(tag: str = PIN_TAG, *, root: Path = Path(".")) -> dict[str, Any]:
    """Refuse (RuntimeError) unless the working tree equals the tag's apart from `RUN_RECORDS`.

    The tag must exist; ``git diff --name-only <tag> HEAD`` may list only run records, and
    ``git status`` may show changed or untracked files only among them (ignored files, such as
    data and checkpoints, are not part of the tree). So a change to the code trees, to a config,
    to pyproject.toml, to a test or to a hash manifest the runners compare inputs against is
    refused. Both run with ``--no-renames``: with rename detection (git's default, or a user's
    ``diff.renames``/``status.renames``) a pinned file moved into reports/ would list only its
    new, allowed path.
    """
    root = Path(root)
    found = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    if found.returncode != 0 or not found.stdout.strip():
        raise RuntimeError(f"the tag {tag} does not exist; the code is not pinned")
    tag_commit = found.stdout.strip()
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {head.stderr.strip()}")
    names = _git(root, "diff", "--name-only", "--no-renames", tag_commit, "HEAD")
    if names.returncode != 0:
        raise RuntimeError(f"git diff against {tag} failed: {names.stderr.strip()}")
    changed = [n for n in names.stdout.splitlines() if n.strip() and not _allowed(n.strip())]
    if changed:
        raise RuntimeError(
            f"HEAD differs from {tag} outside the run records (code trees: "
            f"{', '.join(PINNED_TREES)}): {changed[:10]}"
        )
    status = _git(root, "status", "--porcelain", "--no-renames", "--untracked-files=all")
    if status.returncode != 0:
        raise RuntimeError(f"git status failed: {status.stderr.strip()}")
    dirty = [n for n in _status_paths(status.stdout) if not _allowed(n)]
    if dirty:
        raise RuntimeError(f"uncommitted changes outside the run records: {dirty[:10]}")
    return {"tag": tag, "tag_commit": tag_commit, "head": head.stdout.strip(), "trees": "equal"}
