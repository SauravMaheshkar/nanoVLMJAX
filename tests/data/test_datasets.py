import grain
import pytest

from src.data.datasets import VLMDataset, load_cauldron


@pytest.mark.parametrize(
    "max_samples",
    [1, 2],
    ids=["1-sample", "2-sample"],
)
def test_load_cauldron(max_samples):
    ds = load_cauldron(
        dataset_path="HuggingFaceM4/the_cauldron",
        dataset_name="tqa",
        max_samples=max_samples,
    )
    assert len(ds) == max_samples
    item = ds[0]
    assert "images" in item
    assert "texts" in item


def test_vlm_dataset():
    ds = load_cauldron(
        dataset_path="HuggingFaceM4/the_cauldron",
        dataset_name="tqa",
        max_samples=2,
    )
    assert len(ds) == 2
    item = ds[0]
    assert item is not None


def test_grain_pipeline():
    ds = load_cauldron(
        dataset_path="HuggingFaceM4/the_cauldron",
        dataset_name="tqa",
        max_samples=4,
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-512")
    dataset = VLMDataset(
        hf_dataset=ds,
        image_processor=processor,
    )

    map_ds = grain.MapDataset.source(dataset)
    map_ds = map_ds.filter(lambda x: len(x.get("messages", [])) > 0)

    assert len(map_ds) > 0, "Filter yielded no elements"

    iter_ds = map_ds.to_iter_dataset(
        grain.ReadOptions(num_threads=1, prefetch_buffer_size=10)
    )

    first_item = next(iter(iter_ds))
    assert first_item is not None
    assert "messages" in first_item
