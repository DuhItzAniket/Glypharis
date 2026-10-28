"""The detector API that Phases 10, 17 and 18 call: page image in, text-line boxes out.

`TextLineDetector` chains the three stages every caller must share exactly: `prepare_page` (long
side to ``infer.long_side``, padding to a multiple of 32), a probability-map backend (`TorchProbMap`
here, an exported runtime in Phase 17) and the post-processing at one operating point.
`DetectorSpec` holds every setting those stages read. It is stored in each checkpoint, so a detector
rebuilt from a file behaves exactly as it did in validation.

Two file formats:

- a training checkpoint (`from_checkpoint`), read with ``weights_only=False`` because it holds RNG
  state and plain Python objects: trusted local files only (Phases 8-10). The operating point
  defaults to the checkpoint's ``selected_op``, the operating point of record (spec §7.3);
- a release file (`export_detector`, `from_release`): weights and the spec as tensors and plain
  types only, read with ``weights_only=True``. Phases 17 and 18 load nothing else.
"""

from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from glypharis.config.loader import from_dict
from glypharis.config.schema import (
    DetConfig,
    DetInferConfig,
    DetLabelConfig,
    DetModelConfig,
    DetOperatingPoint,
    DetPostConfig,
)
from glypharis.data.det_labels import LABEL_VERSION
from glypharis.data.download import sha256_file
from glypharis.models.det.dbnet import build_detector
from glypharis.models.det.postprocess import DetBox, boxes_from_candidates, find_candidates
from glypharis.models.det.preprocess import ResizeInfo, prepare_page
from glypharis.models.det.runtime import ProbMapBackend, TorchProbMap
from glypharis.training.checkpoint import load_checkpoint, save_checkpoint

RELEASE_FORMAT: Final = "glypharis-det-release-1"
SPEC_KEYS: Final = ("spec", "detector_spec")  # where a training checkpoint may hold the spec


@dataclass(frozen=True)
class DetectorSpec:
    """Everything inference reads: model shape, label shrink ratio, scale and post-processing."""

    model: DetModelConfig
    labels: DetLabelConfig  # unclip uses labels.shrink_ratio, the r the model was trained with
    infer: DetInferConfig
    post: DetPostConfig
    operating_point: str  # name of one of post.operating_points
    label_version: int

    def __post_init__(self) -> None:
        names = [op.name for op in self.post.operating_points]
        if self.operating_point not in names:
            raise ValueError(f"operating point {self.operating_point!r} is not one of {names}")

    def op(self) -> DetOperatingPoint:
        return next(op for op in self.post.operating_points if op.name == self.operating_point)

    def to_dict(self) -> dict[str, Any]:
        """Plain data only, so a release file stays loadable with ``weights_only=True``."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DetectorSpec:
        return cls(
            model=from_dict(DetModelConfig, d["model"], "spec.model"),
            labels=from_dict(DetLabelConfig, d["labels"], "spec.labels"),
            infer=from_dict(DetInferConfig, d["infer"], "spec.infer"),
            post=from_dict(DetPostConfig, d["post"], "spec.post"),
            operating_point=str(d["operating_point"]),
            label_version=int(d["label_version"]),
        )


def _spec_from_checkpoint(ck: Mapping[str, Any], operating_point: str | None) -> DetectorSpec:
    """The checkpoint's spec at ``operating_point``, else at its ``selected_op``.

    A checkpoint without a stored spec is rebuilt from its resolved ``det`` config section.
    """
    name = operating_point or ck.get("selected_op")
    stored = next((ck[k] for k in SPEC_KEYS if ck.get(k) is not None), None)
    if stored is not None:
        spec = DetectorSpec.from_dict(stored)
        return spec if name is None else dataclasses.replace(spec, operating_point=str(name))
    if name is None:
        raise ValueError("the checkpoint has neither a detector spec nor a selected_op")
    det = from_dict(DetConfig, ck["config"]["det"], "det")
    return DetectorSpec(
        det.model,
        det.labels,
        det.infer,
        det.post,
        str(name),
        int(ck.get("label_version", LABEL_VERSION)),
    )


def _torch_backend(
    spec: DetectorSpec, state_dict: Mapping[str, torch.Tensor], device: str | torch.device
) -> TorchProbMap:
    # Every weight comes from the file, so the ImageNet backbone file is never needed.
    model = build_detector(spec.model, Path("."), load_pretrained=False)
    model.load_state_dict(state_dict)
    return TorchProbMap(model, torch.device(device))


class TextLineDetector:
    """Page image in, `DetBox` list out, at the spec's operating point."""

    def __init__(self, backend: ProbMapBackend, spec: DetectorSpec) -> None:
        self.backend = backend
        self.spec = spec
        self._op = spec.op()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        operating_point: str | None = None,
    ) -> TextLineDetector:
        """A training checkpoint (trusted local file, ``weights_only=False``)."""
        ck = load_checkpoint(path, "cpu")
        spec = _spec_from_checkpoint(ck, operating_point)
        return cls(_torch_backend(spec, ck["model"], device), spec)

    @classmethod
    def from_release(cls, path: str | Path, device: str | torch.device = "cpu") -> TextLineDetector:
        """A release file written by `export_detector`; anything else is refused."""
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as exc:
            raise ValueError(
                f"{path} is not a detector release file (a training checkpoint? write a release "
                f"file with export_detector): {exc}"
            ) from exc
        if not isinstance(obj, dict) or obj.get("format") != RELEASE_FORMAT:
            raise ValueError(f"{path} is not a {RELEASE_FORMAT} file")
        spec = DetectorSpec.from_dict(obj["spec"])
        return cls(_torch_backend(spec, obj["state_dict"], device), spec)

    def prob_map(self, image_rgb: np.ndarray) -> tuple[np.ndarray, ResizeInfo]:
        """The (pad_h, pad_w) float32 probability map of an HWC uint8 RGB page and its geometry."""
        image, info = prepare_page(image_rgb, self.spec.infer.long_side)
        return self.backend(image), info

    def detect(self, image_rgb: np.ndarray) -> list[DetBox]:
        """Text lines of an HWC uint8 RGB page, sorted by (min y, min x)."""
        prob, info = self.prob_map(image_rgb)
        cands, _ = find_candidates(prob, info, self.spec.post)
        boxes, _ = boxes_from_candidates(
            cands, self._op, self.spec.post, self.spec.labels.shrink_ratio, info
        )
        return boxes


def export_detector(checkpoint: Path, out: Path, spec: DetectorSpec | None = None) -> Path:
    """Write the release file for a training checkpoint, at its ``selected_op`` unless ``spec``
    (settings chosen later on validation, such as Phase 9's) replaces the stored one.

    Holds tensors and plain types only, so ``torch.load(out, weights_only=True)`` reads it.
    """
    ck = load_checkpoint(checkpoint, "cpu")
    spec = spec or _spec_from_checkpoint(ck, None)
    config = ck.get("config") or {}
    state = ck.get("state") or {}
    git = ck.get("git") or {}
    payload: dict[str, Any] = {
        "format": RELEASE_FORMAT,
        "state_dict": {k: v.detach().cpu() for k, v in ck["model"].items()},
        "spec": spec.to_dict(),
        "source_sha256": sha256_file(Path(checkpoint)),
        "config_hash": ck.get("config_hash"),
        "experiment_id": config.get("experiment_id"),
        "epoch": state.get("epoch"),
        "git_commit": git.get("commit"),
    }
    return save_checkpoint(Path(out), payload)
