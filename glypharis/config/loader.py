"""YAML → RunConfig: inheritance, overrides, interpolation and strict validation.

Resolution order:
  1. Load the YAML file; recursively load its ``inherits:`` parents (paths relative to the file)
     and deep-merge them, parents first. Mappings merge key by key; lists and scalars replace.
  2. Apply ``key.path=value`` overrides (values parsed as YAML, so ``32`` is an int).
  3. Fill in schema defaults, then resolve ``${dotted.key}`` references inside strings.
  4. Convert to typed dataclasses. Unknown keys, missing required keys and wrong types raise
     ConfigError.
  5. Run value checks (`RunConfig.validate`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from glypharis.config.schema import ConfigError, RunConfig

T = TypeVar("T")

_INTERP_RE = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")
_MAX_INHERIT_DEPTH = 10


# ---------------------------------------------------------------------------- YAML + inheritance


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML error in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``. Nested mappings merge; everything else replaces."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_raw(path: str | Path, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config and its ``inherits:`` parents into one merged dict."""
    path = Path(path).resolve()
    if path in _chain:
        cycle = " -> ".join(p.name for p in (*_chain, path))
        raise ConfigError(f"inheritance cycle: {cycle}")
    if len(_chain) >= _MAX_INHERIT_DEPTH:
        raise ConfigError(f"inheritance deeper than {_MAX_INHERIT_DEPTH} levels at {path}")

    data = _read_yaml(path)
    parents = data.pop("inherits", None)
    if parents is None:
        return data
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ConfigError(f"{path}: 'inherits' must be a path or a list of paths")

    merged: dict[str, Any] = {}
    for parent in parents:
        merged = deep_merge(merged, load_raw(path.parent / parent, (*_chain, path)))
    return deep_merge(merged, data)


# ------------------------------------------------------------------ overrides + interpolation


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides. Values are parsed as YAML scalars/collections."""
    out = json.loads(json.dumps(data))  # deep copy of plain data
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like key.path=value")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        if not all(parts):
            raise ConfigError(f"override {item!r} has an empty key segment")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"override {item!r}: cannot parse value: {exc}") from exc
        node = out
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"override {item!r}: '{part}' is not a section")
            node = child
        node[parts[-1]] = value
    return out


def _lookup(data: Mapping[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"unresolved reference ${{{dotted}}}")
        node = node[part]
    return node


def interpolate(data: dict[str, Any]) -> dict[str, Any]:
    """Replace ``${dotted.key}`` in string values with the referenced (scalar) value."""

    def resolve(value: Any, depth: int = 0) -> Any:
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, str) and "${" in value:
            if depth > 5:
                raise ConfigError(f"interpolation too deep in {value!r}")

            def sub(match: re.Match[str]) -> str:
                target = _lookup(data, match.group(1))
                if isinstance(target, (dict, list)):
                    raise ConfigError(f"${{{match.group(1)}}} refers to a section, not a value")
                if target is None:
                    # Left as-is so partial configs (e.g. base.yaml) still load; entry points
                    # call RunConfig.require_run_identity(), which rejects leftovers.
                    return match.group(0)
                return str(target)

            replaced = _INTERP_RE.sub(sub, value)
            # Recurse only if something changed (references to null values are left in place).
            return replaced if replaced == value else resolve(replaced, depth + 1)
        return value

    resolved = resolve(data)
    assert isinstance(resolved, dict)
    return resolved


# ------------------------------------------------------------------------------ typed conversion


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


def _convert(value: Any, tp: Any, where: str) -> Any:
    origin = get_origin(tp)

    if origin in (Union, types.UnionType):
        options = get_args(tp)
        if value is None:
            if type(None) in options:
                return None
            raise ConfigError(f"{where}: null is not allowed")
        errors = []
        for option in options:
            if option is type(None):
                continue
            try:
                return _convert(value, option, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(errors[0] if len(errors) == 1 else f"{where}: {value!r} matches no type")

    if origin is Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{where}: {value!r} is not one of {list(allowed)}")
        return value

    if origin is dict:
        key_tp, val_tp = get_args(tp)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        return {
            _convert(k, key_tp, f"{where} key {k!r}"): _convert(v, val_tp, f"{where}.{k}")
            for k, v in value.items()
        }

    if origin is list:
        (item_tp,) = get_args(tp)
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        return [_convert(v, item_tp, f"{where}[{i}]") for i, v in enumerate(value)]

    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a section (mapping), got {type(value).__name__}")
        return from_dict(tp, value, where)

    if value is None:
        raise ConfigError(f"{where}: null is not allowed (expected {_type_name(tp)})")
    # bool is a subclass of int in Python; keep them apart so `true` never becomes 1.
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected bool, got {type(value).__name__} {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected int, got {type(value).__name__} {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            hint = ""
            if isinstance(value, str):
                hint = " (YAML reads '1e-3' as text; write 1.0e-3)"
            raise ConfigError(
                f"{where}: expected float, got {type(value).__name__} {value!r}{hint}"
            )
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected str, got {type(value).__name__} {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported schema type {_type_name(tp)}")


def from_dict(cls: type[T], data: Mapping[str, Any], where: str = "") -> T:
    """Build dataclass ``cls`` from a mapping, rejecting unknown keys and wrong types."""
    hints = get_type_hints(cls)
    fields = dataclasses.fields(cls)  # type: ignore[arg-type]
    names = {f.name for f in fields}
    prefix = f"{where}." if where else ""
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(
            "unknown config key(s): "
            + ", ".join(prefix + str(k) for k in unknown)
            + f" (allowed: {', '.join(sorted(names))})"
        )
    # A field without a default (a detector operating point) must be given. Sections merged onto
    # nothing, and list items, never receive defaults, so name the missing key instead of letting
    # cls(**kwargs) raise TypeError.
    missing = sorted(
        f.name
        for f in fields
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
        and f.name not in data
    )
    if missing:
        raise ConfigError("missing config key(s): " + ", ".join(prefix + k for k in missing))
    kwargs = {
        key: _convert(value, hints[key], f"{where}.{key}" if where else key)
        for key, value in data.items()
    }
    return cls(**kwargs)


# ---------------------------------------------------------------------------------- public API


def build_config(data: Mapping[str, Any], overrides: Iterable[str] = ()) -> RunConfig:
    """Overrides → schema defaults → interpolation → typed RunConfig → value validation.

    Defaults are merged in before interpolation so that a default such as
    ``checkpoint.dir: checkpoints/${experiment_id}`` sees values set anywhere in the file.
    Unknown keys survive the merge and are still rejected by `from_dict`.
    """
    user = apply_overrides(dict(data), overrides)
    merged = interpolate(deep_merge(config_to_dict(RunConfig()), user))
    cfg = from_dict(RunConfig, merged)
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> RunConfig:
    """Load, merge, override, interpolate and validate a config file."""
    return build_config(load_raw(path), overrides)


# Fields added in Phase 7, left out of the dict at their defaults: every earlier config
# (GW-REC-001 to 020) keeps its hash and still matches the runs it produced.
_ADDED_DEFAULTS: dict[tuple[str, str], Any] = {
    ("model", "frame_stride"): 4,
    ("model", "svtr"): None,
    ("data", "ctc_aware_batching"): False,
}


def config_to_dict(cfg: RunConfig) -> dict[str, Any]:
    """The resolved config as plain data: what is hashed, saved and compared on resume.

    An absent ``det`` section is left out, and a det run leaves out the recognizer ``data`` and
    ``model`` sections (validated to be defaults). The detector schema therefore cannot move a
    recognizer hash, and recognizer schema growth cannot move a detector hash. Recognizer fields
    added after runs were hashed are left out while they hold their defaults, so older runs keep
    their hashes.
    """
    d = dataclasses.asdict(cfg)
    if d.get("det") is None:
        d.pop("det", None)
    if cfg.task == "det":
        d.pop("data", None)
        d.pop("model", None)
    for (section, key), default in _ADDED_DEFAULTS.items():
        part = d.get(section, {})
        if key in part and part[key] == default:
            del part[key]
    return d


def config_hash(cfg: RunConfig) -> str:
    """Stable SHA-256 of the resolved config (key order independent)."""
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_resolved(cfg: RunConfig, path: str | Path) -> Path:
    """Write the fully resolved config next to a checkpoint or report."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# resolved config, sha256={config_hash(cfg)}\n")
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False, allow_unicode=True)
    return out
