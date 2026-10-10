"""Shared command-line options for every entry point (train, evaluate, scripts).

--config PATH      YAML config (required)
--seed N           shortcut for seed=N
--device D         shortcut for device=D (auto | cuda | cpu)
--resume PATH      shortcut for checkpoint.resume=PATH
--set KEY=VALUE    any other override; repeatable
"""

from __future__ import annotations

import argparse

from glypharis.config.loader import load_config
from glypharis.config.schema import RunConfig


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--seed", type=int, default=None, help="override the random seed")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default=None, help="override the device"
    )
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key, e.g. --set train.batch_size=32 (repeatable)",
    )
    return parser


def overrides_from_args(args: argparse.Namespace) -> list[str]:
    """Shortcut flags first, then --set items, so an explicit --set wins over a shortcut."""
    overrides: list[str] = []
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.device is not None:
        overrides.append(f"device={args.device}")
    if args.resume is not None:
        overrides.append(f"checkpoint.resume={args.resume}")
    overrides.extend(args.overrides)
    return overrides


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return load_config(args.config, overrides_from_args(args))
