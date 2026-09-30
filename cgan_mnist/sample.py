"""Generate class-conditioned PNG grids from a trained checkpoint."""

import argparse
import math
import struct
import zlib
from pathlib import Path

import numpy as np
import torch

from .model import ARCHITECTURE, Generator, LATENT_DIM


def save_grid(images: torch.Tensor, path: Path, columns: int = 10, scale: int = 4) -> None:
    """Write grayscale images in [-1, 1] as a PNG, using only the standard library."""
    if images.ndim != 4 or images.shape[1:] != (1, 16, 16):
        raise ValueError("expected images shaped [N, 1, 16, 16]")
    pixels = images.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).byte().numpy()
    rows = math.ceil(len(pixels) / columns)
    canvas = np.zeros((rows * 16, columns * 16), dtype=np.uint8)
    for index, image in enumerate(pixels):
        row, column = divmod(index, columns)
        canvas[row * 16:(row + 1) * 16, column * 16:(column + 1) * 16] = image[0]
    if scale != 1:
        canvas = np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1)
    height, width = canvas.shape

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
            ">I", zlib.crc32(kind + payload)
        )

    raw = b"".join(b"\x00" + canvas[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def generate_grid(
    generator: Generator, device: torch.device, path: Path, seed: int = 42,
    per_class: int = 10,
) -> None:
    rng = torch.Generator(device=device).manual_seed(seed)
    labels = torch.arange(10, device=device).repeat(per_class)
    noise = torch.randn(len(labels), LATENT_DIM, device=device, generator=rng)
    generator.eval()
    with torch.no_grad():
        images = generator(noise, labels)
    save_grid(images, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a trained cGAN checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/samples.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        parser.error("checkpoint is not for the current CNN; train with python -m cgan_mnist.train")
    model = Generator().to(device)
    model.load_state_dict(checkpoint["generator"])
    generate_grid(model, device, args.output, args.seed, args.per_class)
    print(f"saved {args.output}")
