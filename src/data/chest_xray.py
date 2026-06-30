"""Pipeline Chest X-Ray (Kaggle): NORMAL vs PNEUMONIA."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms

from ..config import (
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    NUM_WORKERS,
    SEED,
    VAL_FRACTION,
)

CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def class_names() -> Tuple[str, ...]:
    return CLASS_NAMES


class ChestXRayBinaryDataset(Dataset):
    """Imagens em pastas train/NORMAL, train/PNEUMONIA (ou caminho único listando ambas)."""

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Pasta do split não encontrada: {split_dir}. "
                "Baixe o dataset do Kaggle e extraia em data/chest_xray/"
            )

        self.samples: list[tuple[Path, int]] = []
        for label_idx, cls in enumerate(CLASS_NAMES):
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Classe ausente: {cls_dir}")
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    self.samples.append((p, label_idx))

        if not self.samples:
            raise RuntimeError(f"Nenhuma imagem encontrada em {split_dir}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _train_val_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if not train_idx:
        train_idx = val_idx[:-1]
        val_idx = val_idx[-1:]
    return train_idx, val_idx


class ChestXRayBinaryDataModule:
    """Monta train/val a partir da pasta train/ e um loader para test/."""

    def __init__(
        self,
        data_root: Path | str,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        val_fraction: float = VAL_FRACTION,
        seed: int = SEED,
    ) -> None:
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

        self.train_tf = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.eval_tf = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        full_train = ChestXRayBinaryDataset(self.data_root, "train", transform=self.train_tf)
        train_idx, val_idx = _train_val_indices(len(full_train), val_fraction, seed)
        self._train_subset = Subset(full_train, train_idx)
        val_base = ChestXRayBinaryDataset(self.data_root, "train", transform=self.eval_tf)
        self._val_subset = Subset(val_base, val_idx)

        self.test_ds = ChestXRayBinaryDataset(self.data_root, "test", transform=self.eval_tf)
        self._train_sampler = self._build_weighted_sampler()

    def _subset_labels(self, subset: Subset) -> list[int]:
        labels: list[int] = []
        for idx in subset.indices:
            _, y = subset.dataset.samples[idx]
            labels.append(int(y))
        return labels

    def train_class_counts(self) -> dict[int, int]:
        labels = self._subset_labels(self._train_subset)
        out = {0: 0, 1: 0}
        for y in labels:
            out[y] += 1
        return out

    def class_weights_tensor(self) -> torch.Tensor:
        counts = self.train_class_counts()
        total = counts[0] + counts[1]
        # peso inversamente proporcional à frequência da classe no treino
        w0 = total / (2.0 * max(1, counts[0]))
        w1 = total / (2.0 * max(1, counts[1]))
        return torch.tensor([w0, w1], dtype=torch.float32)

    def _build_weighted_sampler(self) -> WeightedRandomSampler:
        labels = self._subset_labels(self._train_subset)
        counts = {0: 0, 1: 0}
        for y in labels:
            counts[y] += 1
        sample_weights = [1.0 / max(1, counts[y]) for y in labels]
        weights = torch.as_tensor(sample_weights, dtype=torch.double)
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self._train_subset,
            batch_size=self.batch_size,
            sampler=self._train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self._val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def describe(self) -> str:
        return (
            f"data_root={self.data_root}\n"
            f"train (subset): {len(self._train_subset)} | val: {len(self._val_subset)} | "
            f"test: {len(self.test_ds)}"
        )


def iter_batch_shapes(loader: DataLoader) -> Iterable[tuple[torch.Size, torch.Size]]:
    for x, y in loader:
        yield x.shape, y.shape
        break
