[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SauravMaheshkar/nanoVLMJAX/blob/main/notebooks/nanovlm_tqa.ipynb)
[![CI/CD](https://github.com/SauravMaheshkar/nanoVLMJAX/actions/workflows/ci.yml/badge.svg)](https://github.com/SauravMaheshkar/nanoVLMJAX/actions/workflows/ci.yml)

A minimal, pure JAX implementation of Vision Language Models

### Design Philosophy

* Built on thin abstractions with maximum control, keeping "state" and "functionality" cleanly separated
* Fully accelerator-agnostic — write once, scale to thousands of GPUs/TPUs without touching layer code
* No hidden framework magic — every operation is explicit, debuggable, and portable across hardware

### Setup

```bash
uv sync --all-extras
```

### Getting started

```bash
uv run python main.py --workdir=artifacts/
```

Log training data and model checkpoints to wandb

```bash
uv run python main.py --workdir=artifacts/ \
    --log_wandb true \
    --wandb_project=nanovlmjax \
    --wandb_entity=your-entity
```

Log model checkpoints to huggingface

```bash
uv run python main.py --workdir=artifacts/ \
    --push_to_hub true \
    --repo_id=your-username/nanovlmjax
```

### References

* https://github.com/huggingface/nanoVLM
* https://github.com/jax-ml/jax-llm-examples
* https://github.com/AakashKumarNain/nanoGPTJAX

### Acknowledgements

Thanks to the ML Developer Programs' team at Google for providing GCP credits.
