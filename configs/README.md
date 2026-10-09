# configs/

Every hyperparameter lives in a YAML file here. Code reads values from the validated config object;
it never hard-codes them.

The schema is `glypharis/config/schema.py`; the loader is `glypharis/config/loader.py`. A config file
may `inherits:` one or more parents (paths relative to the file); mappings merge key by key, lists and
scalars replace. Strings may reference other values as `${dotted.key}`. Unknown keys, wrong types and
out-of-range values are errors. `python scripts/validate_configs.py` checks every file here (also in CI);
files under `experiments/` must name an `experiment_id`, `task` and `model_name`.

Every entry point accepts `--config PATH`, `--seed N`, `--device auto|cuda|cpu`, `--resume PATH` and any
number of `--set key.path=value` overrides.

Note: PyYAML reads `1e-3` (no decimal point) as text. Write floats as `1.0e-3`.

## Files

| File | Phase | Purpose |
|---|---|---|
| `base.yaml` | 1 | Shared defaults: seed, device, precision, logging, paths |
| `glypharis_rec.yaml` | 6 | CRNN recognizer baseline (GW-REC-001) |
| `glypharis_rec_svtr.yaml` | 7 | SVTR-style recognizer candidate |
| `glypharis_det.yaml` | 8 | DBNet-style detector baseline |
| `pipeline.yaml` | 10 | Detector + recognizer + reading order + confidence |
| `experiments/GW-*.yaml` | 6+ | One small override file per registered experiment |

Experiment files only contain the keys that differ from their parent, and declare the parent with
`inherits:`. The resolved, fully-expanded config is saved next to every checkpoint.

## Planned schema (training)

```yaml
experiment_id: GW-REC-001        # must exist in research/experiments.md
task: rec                        # rec | det
model_name: glypharis_rec_crnn
seed: 1337
device: auto                     # auto | cuda | cpu   (auto falls back to CPU)

data:
  train: [hiertext_lines_train, glyphsynth_lines_100k]   # names from data/DATASET_MANIFEST.yaml
  val:   [hiertext_lines_val]
  dataset_subset: null           # e.g. 25000 for scaling studies
  charset: data/manifests/charset.json
  image_height: 32
  image_width: null              # null = variable width with bucketing
  max_width: 1024
  num_workers: 6
  augmentation: none             # none | geometric | photometric | full | <named preset>

train:
  epochs: 50
  batch_size: 64
  gradient_accumulation: 1
  optimizer: adamw
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  scheduler: onecycle            # onecycle | cosine | step | plateau
  warmup_steps: 1000
  grad_clip_norm: 5.0
  mixed_precision: bf16          # bf16 | fp16 | none  (auto-fallback if unsupported)

checkpoint:
  dir: checkpoints/${experiment_id}
  checkpoint_interval: 1         # epochs between epoch_XXX.pt snapshots
  keep_last: 3

early_stopping:
  enabled: true
  monitor: val_cer               # never a training metric
  mode: min
  patience: 8
  min_delta: 0.001

diagnostics:
  overfit_window: 5              # epochs used for trend tests
  underfit_cer_threshold: 0.5
  blank_ratio_alarm: 0.95
  grad_norm_alarm: 1.0e3
  stop_on_nan: true
```

The defaults above are placeholders that illustrate the shape of the schema, not tuned values.
