# nanoVLMJAX Agent Instructions

## Quick Start
```bash
uv sync --all-extras           # Install all dependencies
uv run pytest tests/           # Run tests
uv run ruff check src/          # Lint
```

## Project Structure
- `src/data/` - Data loading with Grain + HuggingFace datasets (the_cauldron/tqa)
- `src/models/` - Vision language model components (ViT, language model, modality projector)
- `tests/` - Test files, use `tests/data/` for data module tests

## Key Commands
- Run single test: `uv run pytest tests/data/test_datasets.py::test_load_cauldron -v`
- Run data tests only: `uv run pytest tests/data/ -v`
- Run model tests only: `uv run pytest tests/models/ -v`

## Important Conventions
- **Package manager**: Use `uv` (not pip)
- **Testing**: Tests run via `uv run pytest`, pytest configured in `pyproject.toml`
- **Linting**: `ruff check src/` (auto-fixable with `--fix`)
- **Python path**: Set to `.` in pyproject.toml
- **Never write `__init__.py` files** - keep them empty, exports go in AGENTS.md
- **Imports**: Use `from src.data.` or `from src.models.` pattern

## Coding Style
- **Tensor creation**: Always pass dtype explicitly when creating tensors (e.g., `np.array(..., dtype=np.int32)`)
- **Function parameters**: Use dtype as a parameter with a default value instead of hardcoding

## Test Style
- **Shared testing helpers**: Re-usable test utilities (e.g. mesh context managers, batch sharding helpers, assertion helpers) belong in `tests/conftest.py`, not duplicated across individual test modules.
- **Parameterizing mesh sizes**: Use `pytest.mark.parametrize` inside test functions to vary mesh shapes (e.g. `None`, `1x8`, `2x4`) rather than relying solely on the session-level `mesh` fixture.
- **Mesh coverage**: Every test that exercises distributed behavior should also exercise non-distributed (single-device / no-mesh) behavior in the same function via parameterization.

## Data Module (src/data/)
- Imports: `from src.data.datasets import ...`, `from src.data.loss import ...`, `from src.data.transforms import ...`
- Dataset: `VLMDataset` class, `load_cauldron()` function
- Smallest test dataset: `tqa` from `HuggingFaceM4/the_cauldron` (1496 images)
- Other options: `ai2d`, `aokvqa`, `chart2text`, `chartqa`, `clevr`, `cocoqa`, etc.

## Models Module (src/models/)
- Reference implementation: https://github.com/huggingface/nanoVLM

### Language Model (`language_model.py`)
- Components: RMSNorm, RotaryEmbedding, GroupedQueryAttention (GQA), MLP, Block, LanguageModel
- Supports from_pretrained (e.g., HuggingFaceTB/SmolLM2-135M), save_pretrained, push_to_hub

### Modality Projector (`modality_projector.py`)
- Pixel shuffle + linear projection for vision-to-language space mapping

### Vision Language Model (`vision_language_model.py`)
- Combines ViT + ModalityProjector + LanguageModel
- **Important**: `image_token_id` must be set via config (derived from tokenizer)

## Key Config Parameters
- **ViT**: vit_hidden_dim, vit_inter_dim, vit_patch_size, vit_img_size, vit_n_heads, vit_n_blocks
- **Language Model**: lm_vocab_size, lm_hidden_dim, lm_n_heads, lm_n_kv_heads, lm_n_layers, lm_inter_dim, lm_tie_weights
- **Modality Projector**: mp_pixel_shuffle_factor, mp_image_token_length
- **VLM**: image_token_id (user-provided, from tokenizer)

## Testing Notes
- Memory optimization: Tests use `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` for CI environments to prevent OOM
- **Local-only tests**: Tests that load real datasets, large pretrained models, or otherwise consume too much memory for CI runners should be marked with `@pytest.mark.local`. The CI workflow runs `pytest -m "not local"` to skip them. Run all tests locally with `uv run pytest tests/`.

## Dependencies
- Core: `jax`, `grain`, `datasets`, `transformers`, `optax`, `pillow`
- Dev: `ruff`, `pytest`, `pre-commit`

## References
- Original nanoVLM: https://github.com/huggingface/nanoVLM
- Grain docs: https://google-grain.readthedocs.io/
