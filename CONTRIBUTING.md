# Contributing

Bug reports and reproducibility improvements are welcome. Open an issue describing the model, dataset split, environment, command, and complete error message. Pull requests should remain focused, include tests for changed dependency-free behavior, and pass:

```bash
python -m pytest
ruff check .
```

Do not commit datasets, model weights, credentials, generated predictions, or content whose redistribution terms are unclear.
