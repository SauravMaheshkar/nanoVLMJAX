import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging

from src.input_pipeline import get_dataloaders, iter_batches
from src.models.vision_language_model import (
    VisionLanguageModel,
    VLMConfig,
    vlm_forward,
)
from src.opt_factory import get_optimizer

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def _get_run_name(train_cfg, vlm_cfg):
    """Generate a run name similar to nanoVLM."""
    bs = train_cfg.batch_size * train_cfg.gradient_accumulation_steps
    lr = (
        f"lr_v_{train_cfg.lr_vision_backbone}"
        f"_l_{train_cfg.lr_language_backbone}"
        f"_m_{train_cfg.lr_mp}"
    )
    date = time.strftime("%m%d-%H%M%S")
    vit = vlm_cfg.vit_model_type.split("/")[-1]
    llm = vlm_cfg.lm_model_type.split("/")[-1]
    return f"nanoVLMJAX_{vit}_{llm}_bs{bs}_{lr}_{date}"


def _compute_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    attention_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Masked cross-entropy loss."""
    valid_mask = (labels != -100) & (attention_mask == 1)
    safe_labels = jnp.where(valid_mask, labels, 0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, safe_labels)
    loss = jnp.where(valid_mask, loss, 0.0)
    return loss.sum() / jnp.maximum(valid_mask.sum(), 1.0)


def _loss_fn(
    params,
    batch: dict[str, jnp.ndarray],
    key: jax.Array | None,
    use_mixed_precision: bool,
):
    """Forward pass and loss computation."""
    if use_mixed_precision:
        params = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16)
            if jnp.issubdtype(x.dtype, jnp.floating)
            else x,
            params,
        )

    logits = vlm_forward(
        params,
        batch["input_ids"],
        batch.get("images"),
        key,
        batch.get("attention_mask"),
    )
    logits = logits.astype(jnp.float32)
    loss = _compute_loss(logits, batch["labels"], batch["attention_mask"])
    return loss, logits


@partial(jax.jit, static_argnames=("use_mixed_precision",))
def _compute_grads(params, batch, key, use_mixed_precision: bool):
    """JITted forward/backward pass returning loss and gradients."""
    grad_fn = jax.value_and_grad(
        lambda p: _loss_fn(p, batch, key, use_mixed_precision)[0]
    )
    return grad_fn(params)


def train_step(
    params,
    opt_state,
    batch,
    key,
    optimizer,
    use_mixed_precision: bool,
):
    """Single training step with gradient computation and optimizer update."""
    loss, grads = _compute_grads(params, batch, key, use_mixed_precision)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@partial(jax.jit, static_argnames=("use_mixed_precision",))
def eval_step(
    params,
    batch,
    use_mixed_precision: bool,
):
    """Single evaluation step (no gradients)."""
    loss, _ = _loss_fn(params, batch, None, use_mixed_precision)
    return loss


def train_and_evaluate(
    train_cfg,
    vlm_cfg: VLMConfig,
    workdir: str,
):
    """Main training and evaluation loop."""
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    rng = jax.random.PRNGKey(train_cfg.seed)

    # Initialize or resume model before building the input pipeline so that
    # the loaded checkpoint config (e.g. image_token_id) drives tokenization.
    if train_cfg.resume_from_vlm_checkpoint:
        logging.info("Resuming from checkpoint: %s", vlm_cfg.vlm_checkpoint_path)
        params = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
        # Sync tokenization-related fields from the checkpoint into vlm_cfg so
        # the data pipeline uses IDs that match the loaded model.
        vlm_cfg.image_token_id = params.image_token_id
    else:
        logging.info("Initializing model from scratch")
        # Initialize on a single device to avoid wasting memory across devices
        # when training is not explicitly multi-device.
        single_device = np.array([jax.devices()[0]]).reshape(1, 1)
        mesh = jax.sharding.Mesh(single_device, axis_names=("fsdp", "tp"))

        class ShardingRule:
            batch = "fsdp"
            seq = None
            hidden = "tp"
            tp = "tp"
            kernel_h = None
            kernel_w = None
            in_channels = None
            out_channels = "tp"

        key, init_key = jax.random.split(rng)
        params = VisionLanguageModel.init(init_key, mesh, ShardingRule, vlm_cfg)

    num_params = sum(p.size for p in jax.tree.leaves(params))
    logging.info("Model initialized with %d parameters", num_params)

    # Data loaders (must happen after model init so checkpoint config is respected)
    train_iter, val_iter, tokenizer, tokenize_fn = get_dataloaders(train_cfg, vlm_cfg)

    # Ensure image_token_id is aligned with tokenizer
    if vlm_cfg.image_token_id <= 0:
        raise ValueError(
            f"image_token_id must be > 0, got {vlm_cfg.image_token_id}. "
            "Set it to the tokenizer's image token ID for multimodal training."
        )

    # Optimizer
    optimizer = get_optimizer(
        params,
        lr_mp=train_cfg.lr_mp,
        lr_vision=train_cfg.lr_vision_backbone,
        lr_language=train_cfg.lr_language_backbone,
        max_steps=train_cfg.max_training_steps,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        max_grad_norm=train_cfg.max_grad_norm,
    )
    ms_wrapper = optimizer if hasattr(optimizer, "has_updated") else None

    opt_state = optimizer.init(params)

    # Wandb
    run = None
    if train_cfg.log_wandb and _HAS_WANDB:
        run = wandb.init(
            project=train_cfg.wandb_project,
            entity=train_cfg.wandb_entity,
            config={
                "train": vars(train_cfg),
                "vlm": vars(vlm_cfg),
            },
            name=_get_run_name(train_cfg, vlm_cfg),
        )
        wandb.summary["num_params"] = num_params

    # Training state
    global_step = 0
    best_val_loss = float("inf")
    best_model_path = None
    accumulated_train_loss = 0.0
    train_loss_count = 0

    pad_token_id = (
        int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    )

    while global_step < train_cfg.max_training_steps:
        logging.info("Starting training epoch at global_step %d", global_step)

        max_length = min(train_cfg.max_sample_length, vlm_cfg.lm_max_length)
        train_batch_iter = iter_batches(
            train_iter,
            tokenize_fn,
            batch_size=train_cfg.batch_size,
            pad_token_id=pad_token_id,
            max_length=max_length,
        )

        data_load_start = time.time()
        for batch in train_batch_iter:
            # Convert batch to JAX arrays
            jax_batch = {
                k: jnp.array(v) if isinstance(v, np.ndarray) else v
                for k, v in batch.items()
            }
            data_load_time = time.time() - data_load_start

            rng, step_key = jax.random.split(rng)

            fw_bw_start = time.time()
            params, opt_state, loss = train_step(
                params,
                opt_state,
                jax_batch,
                step_key,
                optimizer,
                train_cfg.use_mixed_precision,
            )
            fw_bw_time = time.time() - fw_bw_start
            loss_value = float(loss)

            accumulated_train_loss += loss_value
            train_loss_count += 1

            # Check if this was a real optimizer update step
            is_update_step = True
            if ms_wrapper is not None:
                is_update_step = bool(ms_wrapper.has_updated(opt_state))

            if is_update_step:
                global_step += 1

                # Logging
                if (
                    global_step % train_cfg.stats_log_interval == 0
                    and train_loss_count > 0
                ):
                    avg_train_loss = accumulated_train_loss / train_loss_count
                    tokens_per_second = float(jnp.sum(jax_batch["attention_mask"])) / (
                        data_load_time + fw_bw_time
                    )
                    logging.info(
                        "Step %d/%d | Loss: %.4f | Tokens/s: %.2f | "
                        "Data: %.3fs | FW/BW: %.3fs",
                        global_step,
                        train_cfg.max_training_steps,
                        avg_train_loss,
                        tokens_per_second,
                        data_load_time,
                        fw_bw_time,
                    )

                    if run is not None:
                        run.log(
                            {
                                "train_loss": avg_train_loss,
                                "tokens_per_second": tokens_per_second,
                                "data_load_time": data_load_time,
                                "fw_bw_time": fw_bw_time,
                            },
                            step=global_step,
                        )

                    accumulated_train_loss = 0.0
                    train_loss_count = 0

                # Evaluation
                if global_step % train_cfg.eval_interval == 0:
                    logging.info("Running validation at step %d", global_step)
                    val_loss_sum = 0.0
                    val_batches = 0

                    val_batch_iter = iter_batches(
                        val_iter,
                        tokenize_fn,
                        batch_size=train_cfg.batch_size,
                        pad_token_id=pad_token_id,
                        max_length=max_length,
                    )

                    for val_batch in val_batch_iter:
                        if val_batches >= 64:
                            break
                        jax_val_batch = {
                            k: jnp.array(v) if isinstance(v, np.ndarray) else v
                            for k, v in val_batch.items()
                        }
                        val_loss = eval_step(
                            params,
                            jax_val_batch,
                            train_cfg.use_mixed_precision,
                        )
                        val_loss_sum += float(val_loss)
                        val_batches += 1

                    avg_val_loss = val_loss_sum / max(val_batches, 1)
                    logging.info("Step %d | Val Loss: %.4f", global_step, avg_val_loss)

                    if run is not None:
                        run.log({"val_loss": avg_val_loss}, step=global_step)

                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_path = os.path.join(
                            workdir, f"best_step_{global_step}"
                        )
                        os.makedirs(best_model_path, exist_ok=True)
                        params.save_pretrained(best_model_path)
                        logging.info("Saved best model to %s", best_model_path)

                # Periodic checkpoint
                if global_step % (train_cfg.eval_interval * 2) == 0:
                    ckpt_path = os.path.join(workdir, f"step_{global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    params.save_pretrained(ckpt_path)
                    logging.info("Saved checkpoint to %s", ckpt_path)

                if global_step >= train_cfg.max_training_steps:
                    break

            data_load_start = time.time()

    # Save final model
    final_path = os.path.join(workdir, "final")
    os.makedirs(final_path, exist_ok=True)
    params.save_pretrained(final_path)
    logging.info("Saved final model to %s", final_path)

    if run is not None:
        run.finish()
