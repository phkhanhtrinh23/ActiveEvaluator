"""Image datasets, class vocabularies and workload transformations.

Layout under `root`:
    torchvision downloads      mnist, usps, svhn, cifar10
    cifar10.1/                 cifar10.1_v6_data.npy, cifar10.1_v6_labels.npy
    CIFAR-10-C/                <corruption>.npy, labels.npy
    imagenet/val/<wnid>/       ImageNet validation set
    imagenetv2-matched-frequency-format-val/<class index>/
    imagenet-r/<wnid>/
    sketch/<wnid>/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
from PIL import Image

DIGITS = [str(i) for i in range(10)]
CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
EXTS = {".jpg", ".jpeg", ".png", ".JPEG"}


@dataclass
class ImageSet:
    """Lazily indexed images with labels in the task's class space."""

    get: Callable[[int], Image.Image]
    labels: np.ndarray
    classes: List[str]
    subset: np.ndarray | None = None  # restrict predictions to these classes

    def __len__(self) -> int:
        return len(self.labels)

    def images(self, idx: Sequence[int]) -> List[Image.Image]:
        return [self.get(int(i)).convert("RGB") for i in idx]

    def sample(self, n: int | None, seed: int = 0) -> "ImageSet":
        if not n or n >= len(self):
            return self
        idx = np.sort(np.random.default_rng(seed).choice(len(self), n, replace=False))
        return ImageSet(lambda i: self.get(int(idx[i])), self.labels[idx], self.classes, self.subset)


def _arrays(images: np.ndarray, labels, classes) -> ImageSet:
    return ImageSet(lambda i: Image.fromarray(np.asarray(images[i])), np.asarray(labels), classes)


def _folder(path: Path, label_of: Callable[[str], int], classes: List[str]) -> ImageSet:
    files = sorted(f for d in sorted(path.iterdir()) if d.is_dir() for f in d.iterdir() if f.suffix in EXTS)
    labels = np.array([label_of(f.parent.name) for f in files])
    return ImageSet(lambda i: Image.open(files[i]), labels, classes)


def _imagenet_info():
    from timm.data import ImageNetInfo

    info = ImageNetInfo()
    wnids = list(info.label_names())
    names = [info.index_to_description(i) for i in range(len(wnids))]
    return {w: i for i, w in enumerate(wnids)}, names


def load(name: str, root: str | Path, split: str = "test") -> ImageSet:
    root = Path(root)
    import torchvision.datasets as tv

    if name == "mnist":
        ds = tv.MNIST(root, train=split == "train", download=True)
        return _arrays(ds.data.numpy(), ds.targets.numpy(), DIGITS)
    if name == "usps":
        ds = tv.USPS(root, train=split == "train", download=True)
        return _arrays(np.asarray(ds.data, dtype=np.uint8), ds.targets, DIGITS)
    if name == "svhn":
        ds = tv.SVHN(root, split="train" if split == "train" else "test", download=True)
        return _arrays(ds.data.transpose(0, 2, 3, 1), ds.labels, DIGITS)
    if name == "cifar10":
        ds = tv.CIFAR10(root, train=split == "train", download=True)
        return _arrays(ds.data, ds.targets, CIFAR10)
    if name == "cifar10.1":
        d = root / "cifar10.1"
        return _arrays(np.load(d / "cifar10.1_v6_data.npy"), np.load(d / "cifar10.1_v6_labels.npy"), CIFAR10)
    if name == "cifar10-c":
        d = root / "CIFAR-10-C"
        labels = np.load(d / "labels.npy")
        parts = [np.load(f, mmap_mode="r") for f in sorted(d.glob("*.npy")) if f.name != "labels.npy"]
        return ImageSet(lambda i: Image.fromarray(np.asarray(parts[i // len(labels)][i % len(labels)])),
                        np.tile(labels, len(parts)), CIFAR10)
    index, names = _imagenet_info()
    if name == "imagenet":
        return _folder(root / "imagenet" / "val", index.__getitem__, names)
    if name == "imagenet-v2":
        return _folder(root / "imagenetv2-matched-frequency-format-val", int, names)
    if name in ("imagenet-r", "imagenet-sketch"):
        ds = _folder(root / ("imagenet-r" if name == "imagenet-r" else "sketch"), index.__getitem__, names)
        ds.subset = np.unique(ds.labels) if name == "imagenet-r" else None
        return ds
    raise KeyError(name)


def split_source(name: str, root: str | Path) -> tuple[ImageSet, ImageSet]:
    """(data the models were trained on, held-out source data the meta-dataset is built from)."""
    if name == "imagenet":
        val = load(name, root)
        train = ImageSet(lambda i: val.get(2 * i), val.labels[::2], val.classes)
        test = ImageSet(lambda i: val.get(2 * i + 1), val.labels[1::2], val.classes)
        return train, test
    return load(name, root, "train"), load(name, root, "test")


def random_transform(seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    return {
        "angle": float(rng.uniform(-30, 30)),
        "brightness": float(rng.uniform(0.5, 1.5)),
        "contrast": float(rng.uniform(0.5, 1.5)),
        "saturation": float(rng.uniform(0.3, 1.7)),
        "hue": float(rng.uniform(-0.1, 0.1)),
        "blur": float(rng.uniform(0, 1.5)),
        "noise": float(rng.uniform(0, 0.08)),
        "grayscale": float(rng.random() < 0.2),
    }


def apply_transform(img: Image.Image, t: Dict[str, float], seed: int) -> Image.Image:
    import torchvision.transforms.functional as TF

    img = TF.rotate(img, t["angle"], fill=0)
    img = TF.adjust_brightness(img, t["brightness"])
    img = TF.adjust_contrast(img, t["contrast"])
    img = TF.adjust_saturation(img, t["saturation"])
    img = TF.adjust_hue(img, t["hue"])
    if t["blur"] > 0.1:
        img = TF.gaussian_blur(img, kernel_size=5, sigma=t["blur"])
    if t["grayscale"]:
        img = TF.to_grayscale(img, num_output_channels=3)
    if t["noise"] > 0:
        a = np.asarray(img, dtype=np.float32) / 255
        a = a + np.random.default_rng(seed).normal(0, t["noise"], a.shape)
        img = Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))
    return img
