import dataclasses


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    # Learning rates per parameter group
    lr_mp: float = 0.005
    lr_vision_backbone: float = 5e-5
    lr_language_backbone: float = 5e-5

    # Training loop
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    max_training_steps: int = 40_000
    eval_interval: int = 500
    stats_log_interval: int = 100

    # Data
    train_dataset_path: str = "HuggingFaceM4/the_cauldron"
    train_dataset_name: str = "tqa"
    val_size: int = 500
    max_sample_length: int = 4096
    max_images_per_example: int = 4
    num_workers: int = 4
    prefetch_buffer_size: int = 100

    # Model
    resume_from_vlm_checkpoint: bool = False
    image_processor_name: str = "google/siglip-base-patch16-224"
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M"

    # Logging
    log_wandb: bool = False
    wandb_project: str = "nanoVLMJAX"
    wandb_entity: str | None = None

    # System
    seed: int = 42
    use_mixed_precision: bool = True

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)
