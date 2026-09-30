"""Download and read the standard MNIST IDX files without torchvision."""

import argparse
import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _valid(path: Path, expected_md5: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_md5


def download_mnist(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in FILES.items():
        target = root / filename
        if _valid(target, expected_md5):
            print(f"verified {target}")
            continue
        temporary = root / (filename + ".part")
        try:
            print(f"downloading {filename}")
            urllib.request.urlretrieve(BASE_URL + filename, temporary)
            if not _valid(temporary, expected_md5):
                raise ValueError(f"MD5 mismatch: {filename}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"verified {target}")


def _read_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        contents = handle.read()
    magic, count, rows, columns = struct.unpack_from(">IIII", contents)
    if magic != 2051 or (rows, columns) != (28, 28):
        raise ValueError(f"invalid MNIST image header: {path}")
    pixels = np.frombuffer(contents, dtype=np.uint8, offset=16)
    if pixels.size != count * rows * columns:
        raise ValueError(f"invalid MNIST image length: {path}")
    return torch.from_numpy(pixels.copy()).reshape(count, 1, rows, columns)


def _read_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        contents = handle.read()
    magic, count = struct.unpack_from(">II", contents)
    if magic != 2049:
        raise ValueError(f"invalid MNIST label header: {path}")
    labels = np.frombuffer(contents, dtype=np.uint8, offset=8)
    if labels.size != count or np.any(labels > 9):
        raise ValueError(f"invalid MNIST labels: {path}")
    return torch.from_numpy(labels.copy()).long()


class MNIST16(Dataset):
    def __init__(self, root: Path, train: bool = True) -> None:
        prefix = "train" if train else "t10k"
        images = _read_images(root / f"{prefix}-images-idx3-ubyte.gz")
        self.labels = _read_labels(root / f"{prefix}-labels-idx1-ubyte.gz")
        if len(images) != len(self.labels):
            raise ValueError("image and label counts differ")
        # One resize for the whole split avoids repeated work in DataLoader workers.
        self.images = F.interpolate(
            images.float().div_(255), size=(16, 16), mode="bilinear", align_corners=False,
            antialias=True,
        ).mul_(2).sub_(1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify MNIST")
    parser.add_argument("--root", type=Path, default=Path("data/mnist"))
    args = parser.parse_args()
    download_mnist(args.root)
    for train in (True, False):
        dataset = MNIST16(args.root, train=train)
        print(f"{'train' if train else 'test'}: {len(dataset)} {tuple(dataset[0][0].shape)}")
