import numpy as np
from transformers import AutoTokenizer


def get_prefix_len(tokenizer: AutoTokenizer) -> int:
    """
    Compute prefix length for loss mask computation during training.

    When computing loss for causal LM, we skip the chat template prefix
    (e.g., <|im_start|>assistant) at the start of assistant messages.
    The prefix_len is used to determine where the actual response begins.

    The sample_string can be any unique string - length doesn't matter.
    """
    sample_string = "xzyvd"
    chat_templated = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": sample_string}],
        tokenize=False,
        add_special_tokens=False,
    )
    random_string_location = chat_templated.find(sample_string)
    if random_string_location == -1:
        msg = "sample_string not found in chat template - template may have changed"
        raise ValueError(msg)
    return len(tokenizer.encode(chat_templated[:random_string_location]))


def prepare_inputs_and_loss_mask(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    prefix_len: int,
    dtype: np.dtype = np.int32,
) -> dict[str, np.ndarray]:
    if not messages:
        return {
            "input_ids": np.array([], dtype=dtype),
            "attention_mask": np.array([], dtype=dtype),
            "loss_mask": np.array([], dtype=bool),
        }

    conv_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
    )

    input_ids = np.array(conv_ids["input_ids"], dtype=dtype)
    attention_mask = np.array(conv_ids["attention_mask"], dtype=dtype)

    total_len = len(input_ids)
    mask = np.zeros(total_len, dtype=dtype)

    cursor = 0
    for msg in messages:
        segment_ids = tokenizer.apply_chat_template(
            [msg], tokenize=True, add_special_tokens=False
        )
        seg_len = len(segment_ids)

        if msg["role"] == "assistant":
            start = cursor + prefix_len
            end = cursor + seg_len
            if start < total_len:
                end = min(end, total_len)
                mask[start:end] = 1

        cursor += seg_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": mask.astype(bool),
    }


def get_labels(
    input_ids: np.ndarray,
    loss_mask: np.ndarray,
) -> np.ndarray:
    labels = input_ids.copy()
    labels = np.roll(labels, -1)  # Shift labels for causal LM
    labels[-1] = -100  # Last token has no target

    labels = np.where(loss_mask, labels, -100)

    return labels


def prepare_training_sample(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    prefix_len: int,
) -> dict[str, np.ndarray]:
    inputs = prepare_inputs_and_loss_mask(tokenizer, messages, prefix_len)
    labels = get_labels(inputs["input_ids"], inputs["loss_mask"])
    inputs["labels"] = labels
    return inputs
