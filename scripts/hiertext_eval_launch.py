"""Start HierText's pinned `eval.py` with Apache Beam's five-minute job limit lifted.

Apache Beam 2.77's DirectRunner hands the pipeline to its Prism runner. Prism's job-state and
message streams carry a gRPC deadline of `--job_server_timeout`, 300 s by default. `eval.py` builds
its own `PipelineOptions` and accepts no extra flags, so any run longer than five minutes is cut
off with DEADLINE_EXCEEDED. A full HierText split takes longer than that on the laptop
(research/failures.md, 2026-11-17).

This launcher adds `--job_server_timeout` to every `PipelineOptions` built from a flag list, then
runs `eval.py` unchanged as `__main__` from the evaluator's folder (the working directory). The
runner, the evaluator's flags and its code are the same as a direct `python eval.py`; only the
deadline differs.

It runs in the evaluator's own environment (`.venv-hiertext`), never the main one:
`scripts/official_hiertext.py` starts it.
"""

import os
import runpy
import sys

from apache_beam.options import pipeline_options

TIMEOUT_FLAGS = ("--job_server_timeout", "--job-server-timeout")
TIMEOUT_S = 7 * 24 * 3600  # far above any evaluation; the deadline only ends a hung run

_init = pipeline_options.PipelineOptions.__init__


def _init_without_limit(self, flags=None, **kwargs):
    # flags=None means "parse sys.argv" to Beam; that case is left alone. The evaluator's own
    # options, and every view of them the runners take, are built from a flag list.
    if flags is not None and "job_server_timeout" not in kwargs:
        flags = list(flags)
        if not any(str(f).split("=")[0] in TIMEOUT_FLAGS for f in flags):
            flags.append(f"{TIMEOUT_FLAGS[0]}={TIMEOUT_S}")
    _init(self, flags, **kwargs)


def main() -> None:
    pipeline_options.PipelineOptions.__init__ = _init_without_limit
    sys.path[0] = os.getcwd()  # eval.py imports its own evaluator/ package, not scripts/
    sys.argv = ["eval.py", *sys.argv[1:]]
    runpy.run_path("eval.py", run_name="__main__")


if __name__ == "__main__":
    main()
