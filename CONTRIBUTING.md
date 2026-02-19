# Contributing

- **Before pushing:** Run `PYTHONPATH=. python -m pytest tests/ -v` and (if you have the model) `python scripts/verify_real_model.py`. Do not commit `model_cache/`, `.env`, API keys, or model weights; see `.gitignore` and the "Before pushing to a remote" section in [readme.md](readme.md).
