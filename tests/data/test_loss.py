import numpy as np

from src.data.loss import (
    get_labels,
    get_prefix_len,
    prepare_inputs_and_loss_mask,
    prepare_training_sample,
)


def test_get_prefix_len_and_prepare_inputs():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

    prefix_len = get_prefix_len(tokenizer)
    assert isinstance(prefix_len, int)
    assert prefix_len > 0

    result = prepare_inputs_and_loss_mask(tokenizer, [], 0)
    assert len(result["input_ids"]) == 0

    messages = [{"role": "user", "content": "Hello"}]
    result = prepare_inputs_and_loss_mask(tokenizer, messages, prefix_len)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "loss_mask" in result


def test_prepare_inputs_and_loss_mask_user_assistant():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    prefix_len = get_prefix_len(tokenizer)
    result = prepare_inputs_and_loss_mask(tokenizer, messages, prefix_len)
    assert "input_ids" in result
    assert "loss_mask" in result
    assert "attention_mask" in result


def test_get_labels_shift():
    input_ids = np.array([1, 2, 3, 4, 5])
    loss_mask = np.array([1, 1, 1, 1, 1], dtype=bool)
    labels = get_labels(input_ids, loss_mask)
    expected = np.array([2, 3, 4, 5, -100])
    np.testing.assert_array_equal(labels, expected)


def test_get_labels_with_mask():
    input_ids = np.array([1, 2, 3, 4, 5])
    loss_mask = np.array([0, 1, 1, 0, 0], dtype=bool)
    labels = get_labels(input_ids, loss_mask)
    expected = np.array([-100, 3, 4, -100, -100])
    np.testing.assert_array_equal(labels, expected)


def test_prepare_training_sample():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    prefix_len = get_prefix_len(tokenizer)
    result = prepare_training_sample(tokenizer, messages, prefix_len)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "loss_mask" in result
    assert "labels" in result
    assert len(result["input_ids"]) > 0
