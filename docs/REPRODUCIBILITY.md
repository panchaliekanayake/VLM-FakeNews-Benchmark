# Reproducibility protocol

## Fixed experimental choices

- Zero-shot inference; no training or dataset-specific adaptation.
- One image and its original caption per sample.
- The same four-decision prompt for all models and datasets.
- Random seed 42 and batch size 1.
- `Finish[ORIGINAL]` and `Finish[MATCH]` map to True.
- `Finish[TEXT REFUTES]`, `Finish[IMAGE REFUTES]`, and `Finish[MISMATCH]` map to Fake.
- Unparseable model output defaults to True and is counted in `invalid_outputs`.
- Macro precision, recall, and F1 use both fixed classes (True and Fake).
- Mean inference time excludes model loading and uses the first 100 generated samples. CUDA is synchronized around every timed generation.

## Environment strategy

The original model repositories have mutually sensitive dependency versions. Use a separate environment for each model and install this repository into every environment with `python -m pip install -e .`. Then follow the official installation instructions linked in `configs/models.json`. BLIP-2 and Qwen-VL run through Transformers; LLaVA, MiniGPT-4, OTTER, and InstructBLIP additionally require their official packages or repository checkouts. Hugging Face checkpoints use the immutable revisions in that configuration file. For an external checkout, use its `release_reference_revision` and pass the path with `--repo-path` where supported.

The release-reference source commits were recorded while preparing this repository; they were not recoverable from every original experiment log. Exact paper-environment reconstruction is therefore not claimed. Each new run records the local external checkout's actual Git commit when `--repo-path` is supplied.

The code does not embed local machine paths. MiniGPT-4 and InstructBLIP receive external checkout locations through `--repo-path`.

## Run one split

```bash
litevlm-fnd run \
  --model blip2-flan-t5-xl \
  --annotations /datasets/MMFakeBench_test/source/MMFakeBench_test.json \
  --data-root /datasets/MMFakeBench_test \
  --output-dir outputs/blip2/MMFakeBench_test
```

## Run the complete dataset matrix for one model

Activate that model's dedicated environment, then provide the five prepared dataset roots:

```bash
scripts/run_model_matrix.sh blip2-flan-t5-xl outputs \
  /datasets/Fakeddit /datasets/IFND /datasets/MMFakeBench \
  /datasets/Weibo /datasets/DriftBench
```

Repeat with each of the six model identifiers. For MiniGPT-4, export `MODEL_PATH` and `REPO_PATH`; for InstructBLIP, export `REPO_PATH`. `GPU_ID` defaults to 0. This produces all 30 model-by-dataset runs while respecting the separate upstream environments.

For MiniGPT-4, pass its evaluation YAML through `--model-path` and the official checkout through `--repo-path`. For InstructBLIP, pass the LAVIS checkout through `--repo-path`.

Every run writes:

- `predictions.json`: model response and ground truth for every sample;
- `metrics.json`: confusion matrix, class-level scores, macro scores, and invalid-output count;
- `run_metadata.json`: checkpoint and repository revisions, generation settings, dataset SHA-256, package versions, Python/platform, CUDA/cuDNN/GPU identifiers, and measured latency.

## Re-score stored predictions

```bash
litevlm-fnd evaluate --predictions outputs/blip2/MMFakeBench_test/predictions.json --output outputs/blip2/MMFakeBench_test/metrics.json
```

## Regenerate the paper figure

```bash
python scripts/generate_paper_assets.py
```

Published numeric results are immutable transcriptions under `results/`; generated run artifacts belong under `outputs/`, which Git ignores. Read [`results/README.md`](../results/README.md) before comparing a rerun with the paper.
