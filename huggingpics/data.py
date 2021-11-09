import logging
import math
import shutil
from io import BytesIO
from pathlib import Path

import pytorch_lightning as pl
import requests
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import ImageFolder
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    Resize, ToTensor)
from transformers import ViTFeatureExtractor

logger = logging.getLogger(__name__)

SEARCH_URL = "https://huggingface.co/api/experimental/images/search"


def get_image_urls_by_term(search_term: str, count=150):
    params = {"q": search_term, "license": "public", "imageType": "photo", "count": count}
    response = requests.get(SEARCH_URL, params=params)
    response.raise_for_status()
    response_data = response.json()
    image_urls = [img['thumbnailUrl'] for img in response_data['value']]
    return image_urls


def gen_images_from_urls(urls):
    num_skipped = 0
    for url in urls:
        response = requests.get(url)
        if not response.status_code == 200:
            num_skipped += 1
        try:
            img = Image.open(BytesIO(response.content))
            yield img
        except UnidentifiedImageError:
            num_skipped += 1

    print(f"Retrieved {len(urls) - num_skipped} images. Skipped {num_skipped}.")


def urls_to_image_folder(urls, save_directory):
    for i, image in enumerate(gen_images_from_urls(urls)):
        image.save(save_directory / f'{i}.jpg')


def make_huggingpics_imagefolder(data_dir, search_terms, count=150, overwrite=False, transform=None):

    data_dir = Path(data_dir)

    if data_dir.exists():
        if overwrite:
            logger.warning(f"Deleting existing HuggingPics data directory to create new one: {data_dir}")
            shutil.rmtree(data_dir)
        else:
            logger.warning(f"Using existing HuggingPics data directory: '{data_dir}'")
            return ImageFolder(str(data_dir), transform=transform)

    for search_term in search_terms:
        search_term_dir = data_dir / search_term
        search_term_dir.mkdir(exist_ok=True, parents=True)
        urls = get_image_urls_by_term(search_term, count)
        logger.info(f"Saving images of {search_term} to {str(search_term_dir)}...")
        urls_to_image_folder(urls, search_term_dir)

    return ImageFolder(str(data_dir), transform=transform)


class HuggingPicsData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        search_terms,
        model_name_or_path='google/vit-base-patch16-224-in21k',
        count=150,
        val_split_pct=0.15,
        batch_size=16,
        num_workers=0,
        pin_memory=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        ds = make_huggingpics_imagefolder(self.hparams.data_dir, self.hparams.search_terms, self.hparams.count)

        classes = ds.classes
        self.num_labels = len(classes)
        self.id2label = {str(i): label for i, label in enumerate(classes)}
        self.label2id = {label: str(i) for i, label in enumerate(classes)}

        feature_extractor = ViTFeatureExtractor.from_pretrained(self.hparams.model_name_or_path)
        normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        self.train_transform = Compose(
            [
                RandomResizedCrop(feature_extractor.size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        self.val_transform = Compose(
            [
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                normalize,
            ]
        )

        indices = torch.randperm(len(ds)).tolist()
        n_val = math.floor(len(indices) * self.hparams.val_split_pct)
        self.train_ds = SubsetWithTransform(ds, indices[:-n_val], transform=self.train_transform)
        self.val_ds = SubsetWithTransform(ds, indices[-n_val:], transform=self.val_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        imgs = torch.stack([ex[0] for ex in batch])
        labels = torch.LongTensor([ex[1] for ex in batch])
        return {'pixel_values': imgs, 'labels': labels}


class SubsetWithTransform(torch.utils.data.Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        img = self.transform(img)
        return img, label
