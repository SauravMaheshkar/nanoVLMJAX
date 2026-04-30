from typing import Any

import grain
import numpy as np
from transformers import AutoProcessor, AutoTokenizer

from src.data.datasets import VLMDataset, load_cauldron
from src.data.loss import get_prefix_len, prepare_training_sample


def build_grain_pipeline(
    dataset: VLMDataset,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 4,
    prefetch_buffer_size: int = 100,
):
    """Build a Grain iterator that yields individual processed samples."""
    map_dataset = grain.MapDataset.source(dataset)

    if shuffle:
        map_dataset = map_dataset.shuffle(seed=seed)

    map_dataset = map_dataset.filter(lambda x: len(x.get("messages", [])) > 0)

    iter_dataset = map_dataset.to_iter_dataset(
        grain.ReadOptions(
            num_threads=num_workers, prefetch_buffer_size=prefetch_buffer_size
        )
    )

    return iter_dataset


def collate_vlm_batch(
    samples: list[dict[str, Any]],
    pad_token_id: int,
    max_length: int | None = None,
) -> dict[str, Any] | None:
    """Collate and left-pad a list of VLM samples into a batched dict of arrays."""
    if not samples:
        return None

    samples = [s for s in samples if s is not None]
    if not samples:
        return None

    if max_length is not None:
        samples = [s for s in samples if len(s["input_ids"]) <= max_length]
        if not samples:
            return None
        max_len = max_length
    else:
        max_len = max(len(s["input_ids"]) for s in samples)

    batch_size = len(samples)
    input_ids = np.full((batch_size, max_len), pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)
    labels = np.full((batch_size, max_len), -100, dtype=np.int32)
    images = []

    for i, sample in enumerate(samples):
        seq_len = len(sample["input_ids"])
        pad_left = max_len - seq_len
        input_ids[i, pad_left:] = sample["input_ids"]
        attention_mask[i, pad_left:] = sample["attention_mask"]
        labels[i, pad_left:] = sample["labels"]

        sample_images = sample.get("images", [])
        if sample_images:
            # Extract pixel_values from processor output dict and squeeze batch dim
            first_img = sample_images[0]
            if isinstance(first_img, dict):
                pixel_values = first_img["pixel_values"]
            else:
                pixel_values = first_img
            if pixel_values.ndim == 4:
                pixel_values = pixel_values[0]
            images.append(pixel_values)
        else:
            # No images for this sample; use zeros
            # (will be ignored by image_token_id mask)
            # Infer image shape from other samples if available
            images.append(None)

    # Only include images if at least one sample has a real image
    has_images = any(img is not None for img in images)
    if has_images:
        target_shape = None
        for img in images:
            if img is not None:
                target_shape = img.shape
                break
        for i in range(len(images)):
            if images[i] is None:
                images[i] = np.zeros(target_shape, dtype=np.float32)
        images = np.stack(images, axis=0)  # (B, C, H, W)
    else:
        images = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": images,
    }


def get_dataloaders(
    train_cfg,
    vlm_cfg,
):
    """Build training and validation Grain pipelines and tokenize them."""
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prefix_len = get_prefix_len(tokenizer)

    # Load the full dataset once and create a non-overlapping train/val split
    full_hf_dataset = load_cauldron(
        train_cfg.train_dataset_path,
        train_cfg.train_dataset_name,
        split="train",
    )
    total_len = len(full_hf_dataset)
    val_size = min(train_cfg.val_size, total_len // 2)

    processor = AutoProcessor.from_pretrained(train_cfg.image_processor_name)
    val_hf = full_hf_dataset.select(range(val_size))
    train_hf = full_hf_dataset.select(range(val_size, total_len))
    train_dataset = VLMDataset(train_hf, processor)
    val_dataset = VLMDataset(val_hf, processor)

    train_iter = build_grain_pipeline(
        train_dataset,
        shuffle=True,
        seed=train_cfg.seed,
        num_workers=train_cfg.num_workers,
        prefetch_buffer_size=train_cfg.prefetch_buffer_size,
    )

    val_iter = build_grain_pipeline(
        val_dataset,
        shuffle=False,
        seed=train_cfg.seed,
        num_workers=max(1, train_cfg.num_workers // 2),
        prefetch_buffer_size=train_cfg.prefetch_buffer_size,
    )

    image_token_id = vlm_cfg.image_token_id
    mp_image_token_length = vlm_cfg.mp_image_token_length

    def _tokenize_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
        """Tokenize messages and attach image data."""
        messages = sample.get("messages", [])
        if not messages:
            return None
        try:
            inputs = prepare_training_sample(tokenizer, messages, prefix_len)
        except Exception:
            return None

        # Prepend image tokens for the first image in the sample
        images = sample.get("images", [])
        if images and image_token_id is not None and image_token_id > 0:
            num_image_tokens = mp_image_token_length
            image_ids = np.full(num_image_tokens, image_token_id, dtype=np.int32)
            image_mask = np.ones(num_image_tokens, dtype=np.int32)
            image_labels = np.full(num_image_tokens, -100, dtype=np.int32)

            inputs["input_ids"] = np.concatenate([image_ids, inputs["input_ids"]])
            inputs["attention_mask"] = np.concatenate(
                [image_mask, inputs["attention_mask"]]
            )
            inputs["labels"] = np.concatenate([image_labels, inputs["labels"]])

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
            "images": images,
        }

    return train_iter, val_iter, tokenizer, _tokenize_sample


def iter_batches(
    grain_iter,
    tokenize_fn,
    batch_size: int,
    pad_token_id: int,
    max_length: int | None = None,
):
    """Yield padded batches from a Grain iterator."""
    batch = []
    for sample in grain_iter:
        processed = tokenize_fn(sample)
        if processed is None:
            continue
        batch.append(processed)
        if len(batch) == batch_size:
            collated = collate_vlm_batch(batch, pad_token_id, max_length)
            if collated is not None:
                yield collated
            batch = []
    if batch:
        collated = collate_vlm_batch(batch, pad_token_id, max_length)
        if collated is not None:
            yield collated
