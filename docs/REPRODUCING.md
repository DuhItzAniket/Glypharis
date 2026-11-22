# Reproducing GLYPHARIS from a fresh clone

The order of work, from nothing to an evaluated model. Commands are for Git Bash on Windows; on Linux or
macOS use `python3 -m venv` and `.venv/bin/python`. `docs/FINAL_VALIDATION.md` records what happened when
these steps were followed on a clean folder.

Python 3.11, at least 3.11.4, is required (`requires-python = ">=3.11.4,<3.12"`, Phase 23 D13). A CUDA build of PyTorch is shown; for CPU only, use
`--index-url https://download.pytorch.org/whl/cpu`.

## 1. Clone and install

```bash
git clone https://github.com/DuhItzAniket/Glypharis.git
cd Glypharis
py -3.11 -m venv .venv
P=.venv/Scripts/python
$P -m pip install --upgrade pip
$P -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
$P -m pip install -r requirements/lock.txt
$P -m pip install -e . --no-deps
$P -m pip check
$P -m pytest -q
$P scripts/validate_configs.py
```

## 2. Download the data

`data/sources.yaml` lists every dataset with its source, size, licence and hashes. Read
[DATASETS.md](../DATASETS.md) first: some sets are for evaluation only.

```bash
DS="--dataset hiertext --dataset imagenet_backbones --dataset funsd --dataset iiit5k"
$P scripts/download_datasets.py $DS --dry-run
$P scripts/download_datasets.py --dataset hiertext --parallel 3
$P scripts/download_datasets.py --dataset imagenet_backbones --dataset funsd --dataset iiit5k
$P scripts/download_datasets.py $DS --verify
```

An interrupted download keeps its partial file; run the same command again and it resumes. Every file
must match its recorded hash. Downloading updates the time stamps in `data/manifests`; nothing else there
should change, and `git checkout -- data/manifests data/DATASET_MANIFEST.yaml` restores them.

GDC and GDC-scan cannot be downloaded: they are the project's own captures, and GDC-scan is private.

## 3. Prepare the data

```bash
$P scripts/prepare_hiertext.py --workers 12
$P scripts/build_eval_crops.py --eval-set hiertext_val --expect-count 74500
$P scripts/prepare_benchmarks.py funsd
$P scripts/prepare_benchmarks.py iiit5k
$P scripts/build_eval_crops.py --eval-set funsd_test --expect-count 2647
git diff --exit-code -- data/manifests/processed
```

The last command passes when the prepared files match the recorded ones. `prepare_hiertext.py` also
rewrites its generated reports; add `--reports <folder>` to send them elsewhere.

## 4. Obtain the model

The v1 weights are not in the repository. They are the asset `glypharis-v1.tar.gz` of the GitHub
release `v1.0`, and `glypharis.download_model` fetches it, checks the archive's size and SHA-256 and the
bundle's manifest SHA-256 against the pins in `glypharis/hub.py`, and installs the bundle (a directory
holding `manifest.json`, two release files and two ONNX files) in the first `GLYPHARIS_MODELS` entry:

```bash
mkdir -p models
export GLYPHARIS_MODELS="$PWD/models"
$P -c "import glypharis; print(glypharis.download_model('glypharis-v1'))"
```

`glypharis-ocr` in step 5 would also fetch it by itself, unless `--no-download` is given or
`GLYPHARIS_OFFLINE` is set. To install it by hand instead, download `glypharis-v1.tar.gz` and
`glypharis-v1.tar.gz.sha256` from the release page, check them with
`sha256sum -c glypharis-v1.tar.gz.sha256`, extract the archive into `models/` (it holds the one folder
`glypharis-v1/`), and check that `sha256sum models/glypharis-v1/manifest.json` gives
`875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037` (GW-EFF-006). `glypharis-check`
then checks the installed bundle against the registry and reads its sample page.

The weights are distributed under CC BY-SA 4.0 with attribution to HierText and Open Images
([ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md)), with the notices in the archive's
`LICENSE-WEIGHTS` and `NOTICE.md`. The detector was initialised from ImageNet weights, a disclosed
exception to the project's commercial-use rule ([ADR-0009](decisions/ADR-0009-no-synthtext.md)), and
the weights are not described as cleared for commercial use.

## 5. Read a page

```bash
.venv/Scripts/glypharis-ocr --image page.jpg --model glypharis-v1 --output results
```

The default backend is PyTorch; add `--backend onnx` for ONNX Runtime, which is slower on the CPU. Each
image gives a JSON and a text file in `results/`.

## 6. Evaluate

`scripts/package_check.py` reads the 400 Phase 5 validation pages with the installed package and compares
them with a reference prediction file (`--phase10`). The reference used for v1 is GW-E2E-006's, which is
not in the repository.

```bash
$P scripts/package_check.py --config configs/det/gw-det-001.yaml --model glypharis-v1 \
  --backend torch --threads 10 --v1 --phase10 <reference>.jsonl --out <report>.json
```

It passes when page CER differs from the reference by at most 0.05 points.
