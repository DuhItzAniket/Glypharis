# data/

Datasets are **never** committed. Only small text records that make the data reproducible live in git.

```
data/
  raw/          # downloads, untouched                      (ignored)
  interim/      # converted annotations, extracted subsets  (ignored)
  processed/    # LMDB / shards used for training           (ignored)
  external/     # external benchmarks                       (ignored)
  cache/        # anything regenerable                      (ignored)
  manifests/    # per-dataset manifests, split files, charset, exclusion lists  (committed)
  DATASET_MANIFEST.yaml   # one record per dataset (committed, Phase 2)
```

Download with `scripts/download_datasets.py` (Phase 2). Provenance and licences: [DATASETS.md](../DATASETS.md).
