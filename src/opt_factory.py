from typing import Callable

import optax
from jax.tree_util import tree_map_with_path


def _partition_fn(path, value):
    """Partition parameters into MP / vision / language groups."""
    keys = [getattr(p, "name", getattr(p, "key", str(p))) for p in path]
    if "modality_projector" in keys:
        return "mp"
    elif "vision_encoder" in keys:
        return "vision"
    else:
        return "language"


def get_lr_schedule(
    max_lr: float, max_steps: int, warmup_ratio: float = 0.03
) -> Callable:
    """Cosine learning rate schedule with linear warmup.

    Mirrors the schedule from Karpathy's nanoGPT / nanoVLM.
    """
    min_lr = max_lr * 0.1
    warmup_steps = int(max_steps * warmup_ratio)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=max_lr,
        warmup_steps=warmup_steps,
        decay_steps=max_steps,
        end_value=min_lr,
    )


def get_optimizer(
    params,
    lr_mp: float,
    lr_vision: float,
    lr_language: float,
    max_steps: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
):
    """Build an AdamW optimizer with per-group LR schedules and gradient accumulation.

    Args:
        params: Model parameter pytree (used to infer partition labels).
        lr_mp: Peak learning rate for the modality projector.
        lr_vision: Peak learning rate for the vision encoder.
        lr_language: Peak learning rate for the language model.
        max_steps: Total number of optimizer update steps.
        gradient_accumulation_steps: Number of mini-steps to accumulate before updating.
        max_grad_norm: Maximum gradient norm for clipping (None disables clipping).
        b1: Adam beta1.
        b2: Adam beta2.
        eps: Adam epsilon.
        weight_decay: Weight decay coefficient.

    Returns:
        An optax gradient transformation (possibly wrapped in MultiSteps).
    """
    mp_schedule = get_lr_schedule(lr_mp, max_steps)
    vision_schedule = get_lr_schedule(lr_vision, max_steps)
    language_schedule = get_lr_schedule(lr_language, max_steps)

    def _make_tx(schedule):
        tx = [
            optax.clip_by_global_norm(max_grad_norm)
            if max_grad_norm is not None
            else None,
            optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_learning_rate(schedule),
        ]
        tx = [t for t in tx if t is not None]
        return optax.chain(*tx)

    transforms = {
        "mp": _make_tx(mp_schedule),
        "vision": _make_tx(vision_schedule),
        "language": _make_tx(language_schedule),
    }

    labels = tree_map_with_path(_partition_fn, params)
    base_optimizer = optax.multi_transform(transforms, labels)

    if gradient_accumulation_steps > 1:
        return optax.MultiSteps(
            base_optimizer,
            every_k_schedule=gradient_accumulation_steps,
            use_grad_mean=True,
        )

    return base_optimizer
