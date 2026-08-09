# Dataset acquisition and normalization

Dataset content is not redistributed. Obtain it from the original providers and accept each provider's terms before using this code.

| Dataset | Primary source | Paper test split | Binary mapping |
|---|---|---:|---|
| Fakeddit | [official repository](https://github.com/entitize/Fakeddit) | 39,494 | `2_way_label=1` → True; `0` → Fake |
| IFND | [authors' dataset page](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset) and [paper](https://doi.org/10.1007/s40747-021-00552-1) | 22,702 | `TRUE` → True; `Fake` → Fake |
| MMFakeBench | [gated official dataset](https://huggingface.co/datasets/liuxuannan/MMFakeBench) | 10,000 | use `gt_answers` directly |
| Weibo | [EANN repository](https://github.com/yaqingwang/EANN-KDD18) | 1,465 | `nonrumor` → True; `rumor` → Fake |
| DriftBench | [official repository](https://github.com/fanxiao15/DriftBench) | 8,120 | `Real` → True; `Fake` → Fake |

## Exact split identity

[`paper_test_splits.csv`](../data/paper_test_splits.csv) records the expected filename, class counts, and SHA-256 digest of each normalized annotation manifest used for the reported test results. Run `sha256sum` on a prepared manifest and compare it with that file. This identifies the exact evaluated rows without republishing restricted captions or image metadata.

The normalized files are JSON lists with the schema in [`data/README.md`](../data/README.md). Preserve the original provider's test membership and row order; do not resample. Add these fields while retaining any source metadata:

```text
text         original post title, caption, or article text
image_path   image location relative to the dataset root
gt_answers   True or Fake after applying the mapping above
fake_cls     original for True; source subtype when supplied, otherwise textual_veracity_distortion
```

For Fakeddit, keep only rows in the provider's multimodal test split whose images are available, producing the counts and digest above. For IFND, retain the published rows with locally available images and the study's held-out test membership. For Weibo, use the EANN `test_nonrumor.txt` and `test_rumor.txt` membership. MMFakeBench already supplies the required fields in `MMFakeBench_test.json`. For DriftBench, combine the official test categories used by the paper and map the provider's `Real`/`Fake` label.

The manifests are intentionally not included because MMFakeBench's gated terms prohibit further publication or distribution of dataset portions or derived data. Users are responsible for checking equivalent restrictions on the other datasets.

Important limitation: the public release alone does not reconstruct the exact normalized Fakeddit, IFND, Weibo, or DriftBench membership. In particular, locally available-image filtering and the held-out split state are not fully represented by the upstream datasets. The committed counts and hashes verify an authorized collaborator's copy but cannot create it. This limitation is stated explicitly instead of publishing restricted manifests or presenting an approximate converter as exact. MMFakeBench's provider-supplied test manifest is directly obtainable after access approval.

## Image layout

Images may use any internal directory layout because every manifest stores its relative path. A typical prepared dataset is:

```text
/datasets/Fakeddit/
├── source/Fakeddit_test.json
└── images/test_set/*.jpg
```

Pass the manifest itself to `--annotations` and the directory against which `image_path` is resolved to `--data-root`. The validator rejects missing images and paths that escape the data root.
