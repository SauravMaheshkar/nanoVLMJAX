from typing import Any

from datasets import Dataset, load_dataset
from grain import MapDataset, ReadOptions
from grain.sources import RandomAccessDataSource
from PIL import Image
from transformers import AutoProcessor


class VLMDataset(RandomAccessDataSource):
    def __init__(
        self,
        hf_dataset: Dataset,
        image_processor: AutoProcessor,
    ):
        self._hf_dataset = hf_dataset
        self._image_processor = image_processor

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._hf_dataset[idx]
        return self._process_item(item)

    def _process_item(self, item: dict[str, Any]) -> dict[str, Any]:
        images = item.get("images")
        texts = item.get("texts", [])

        processed_images = []
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for img in images:
                if isinstance(img, str) or not isinstance(img, Image.Image):
                    with Image.open(img) as opened_image:
                        rgb_image = opened_image.convert("RGB")
                else:
                    rgb_image = img.convert("RGB")
                processed_img = self._image_processor(rgb_image, return_tensors="np")
                processed_images.append(processed_img)

        messages = []
        for text in texts:
            user_msg = text.get("user", "")
            assistant_msg = text.get("assistant", "")
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

        return {
            "images": processed_images,
            "messages": messages,
        }


def load_cauldron(
    dataset_path: str,
    dataset_name: str,
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    hf_dataset = load_dataset(dataset_path, dataset_name, split=split)
    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(max_samples))
    return hf_dataset


def create_grain_pipeline(
    dataset: VLMDataset,
    batch_size: int = 2,
    shuffle: bool = True,
    seed: int = 42,
    num_threads: int = 4,
    prefetch_buffer_size: int = 100,
):
    map_dataset = MapDataset.source(dataset)

    if shuffle:
        map_dataset = map_dataset.shuffle(seed=seed)

    map_dataset = map_dataset.filter(lambda x: len(x.get("messages", [])) > 0)

    iter_dataset = map_dataset.to_iter_dataset(
        ReadOptions(num_threads=num_threads, prefetch_buffer_size=prefetch_buffer_size)
    )

    return iter_dataset.batch(batch_size=batch_size)
