import argparse

from absl import app, logging

from configs.default import TrainConfig
from src.models.vision_language_model import VLMConfig
from train import train_and_evaluate


def _bool_from_string(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")


def main(argv):
    parser = argparse.ArgumentParser(description="Train nanoVLMJAX")
    parser.add_argument(
        "--workdir", required=True, help="Directory to store checkpoints and logs."
    )
    parser.add_argument(
        "--lr_mp", type=float, help="Learning rate for the modality projector."
    )
    parser.add_argument(
        "--lr_vision_backbone",
        type=float,
        help="Learning rate for the vision backbone.",
    )
    parser.add_argument(
        "--lr_language_backbone",
        type=float,
        help="Learning rate for the language backbone.",
    )
    parser.add_argument("--batch_size", type=int, help="Per-device batch size.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--max_training_steps", type=int, help="Maximum training steps."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, help="Maximum gradient norm for clipping."
    )
    parser.add_argument(
        "--eval_interval", type=int, help="Evaluation interval in steps."
    )
    parser.add_argument(
        "--train_dataset_path", type=str, help="HuggingFace dataset path."
    )
    parser.add_argument(
        "--train_dataset_name", type=str, help="HuggingFace dataset config name."
    )
    parser.add_argument("--val_size", type=int, help="Number of validation samples.")
    parser.add_argument("--vlm_checkpoint_path", type=str, help="Checkpoint directory.")
    parser.add_argument(
        "--resume_from_vlm_checkpoint",
        type=_bool_from_string,
        help="Resume from checkpoint.",
    )
    parser.add_argument(
        "--log_wandb",
        type=_bool_from_string,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb_project", type=str, help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity name.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--use_mixed_precision",
        type=_bool_from_string,
        help="Use bfloat16 mixed precision.",
    )
    parser.add_argument(
        "--image_token_id", type=int, help="Image token ID (overrides VLMConfig)."
    )
    parser.add_argument(
        "--image_processor_name", type=str, help="Image processor model name."
    )
    parser.add_argument("--tokenizer_name", type=str, help="Tokenizer model name.")

    args = parser.parse_args(argv[1:])

    train_cfg = TrainConfig()
    vlm_cfg = VLMConfig()

    def _apply_overrides(config, fields):
        for field in fields:
            value = getattr(args, field, None)
            if value is not None:
                config = config.replace(**{field: value})
        return config

    train_cfg = _apply_overrides(
        train_cfg,
        [
            "lr_mp",
            "lr_vision_backbone",
            "lr_language_backbone",
            "batch_size",
            "gradient_accumulation_steps",
            "max_training_steps",
            "max_grad_norm",
            "eval_interval",
            "train_dataset_path",
            "train_dataset_name",
            "val_size",
            "resume_from_vlm_checkpoint",
            "log_wandb",
            "wandb_project",
            "wandb_entity",
            "seed",
            "use_mixed_precision",
            "image_processor_name",
            "tokenizer_name",
        ],
    )

    vlm_cfg = _apply_overrides(
        vlm_cfg,
        [
            "vlm_checkpoint_path",
            "image_token_id",
        ],
    )

    if args.resume_from_vlm_checkpoint:
        train_cfg = train_cfg.replace(resume_from_vlm_checkpoint=True)
        # When resuming a full VLM we don't need to load individual backbone weights
        vlm_cfg = vlm_cfg.replace(vlm_load_backbone_weights=False)

    logging.info("--- VLM Config ---")
    logging.info(vlm_cfg)
    logging.info("--- Train Config ---")
    logging.info(train_cfg)

    train_and_evaluate(train_cfg, vlm_cfg, args.workdir)


if __name__ == "__main__":
    app.run(main)
