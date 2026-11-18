# Reproducing GLYPHARIS from a fresh clone

The order of work, from nothing to an evaluated model. Commands are for Git Bash on Windows; on Linux or
macOS use `python3 -m venv` and `.venv/bin/python`. `docs/FINAL_VALIDATION.md` records what happened when
these steps were followed on a clean folder.

Python 3.11 is required. A CUDA build of PyTorch is shown; for CPU only, use
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

No weights are distributed yet, so this step cannot be done from the repository alone. A release is
Phase 23's decision, under the terms in [ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md). With a
copy of the bundle, which is a directory holding `manifest.json`, two release files and two ONNX files, put
it in a model directory:

```bash
mkdir -p models
cp -r <path-to>/glypharis-v1 models/
export GLYPHARIS_MODELS="$PWD/models"
```

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
