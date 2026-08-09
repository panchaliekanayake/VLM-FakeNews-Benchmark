# Dataset placement

Datasets are not redistributed in this repository. Acquisition links, label mappings, expected split counts, and annotation checksums are in [`docs/DATASETS.md`](../docs/DATASETS.md).

The runner accepts a JSON list with the schema used by MMFakeBench:

```json
[
  {
    "text": "A news caption",
    "image_path": "/images/example.png",
    "gt_answers": "Fake",
    "fake_cls": "mismatch"
  }
]
```

`text`, `image_path`, and `gt_answers` are required. `fake_cls` is retained in output when available. A leading slash in `image_path` is interpreted relative to `--data-root`; it is not treated as a filesystem-root path.

Before inference, validate a split:

```bash
litevlm-fnd validate-data --annotations /path/to/split.json --data-root /path/to/dataset
```

The five evaluated test splits contain 39,494 Fakeddit, 22,702 IFND, 10,000 MMFakeBench, 1,465 Weibo, and 8,120 DriftBench samples, as reported in the paper.
