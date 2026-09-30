"""Train the compact cGAN on 16x16 MNIST."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import MNIST16
from .model import ARCHITECTURE, Discriminator, Generator, LATENT_DIM, NUM_CLASSES, parameter_count
from .sample import generate_grid


def describe() -> None:
    generator, discriminator = Generator(), Discriminator()
    labels = torch.tensor([0, 1])

    def run_with_shapes(model: nn.Module, *inputs: torch.Tensor):
        handles = []
        for name, layer in model.named_modules():
            if not name or list(layer.children()):
                continue

            def show(_layer: nn.Module, _inputs: tuple, output: torch.Tensor,
                     name: str = name) -> None:
                count = sum(p.numel() for p in _layer.parameters(recurse=False))
                print(f"  {name:<20} {str(tuple(output.shape)):<18} {count:>7,}")

            handles.append(layer.register_forward_hook(show))
        try:
            with torch.no_grad():
                return model(*inputs)
        finally:
            for handle in handles:
                handle.remove()

    print("Generator (batch size 2):")
    fake = run_with_shapes(generator, torch.randn(2, LATENT_DIM), labels)
    print(f"  output               {tuple(fake.shape)}; {parameter_count(generator):,} parameters")
    print("Discriminator (batch size 2):")
    logits, code_logits = run_with_shapes(discriminator, fake, labels)
    print(f"  output               real/fake {tuple(logits.shape)}, "
          f"code {tuple(code_logits.shape)}; {parameter_count(discriminator):,} parameters")
    print(f"Total: {parameter_count(generator) + parameter_count(discriminator):,} parameters")


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    dataset = MNIST16(args.data, train=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    code_criterion = nn.CrossEntropyLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        sum_d = sum_g = sum_q_real = sum_q_fake = 0.0
        sum_match = sum_mismatch = 0.0
        correct_q_real = correct_q_fake = examples = 0
        steps = 0
        for real, labels in loader:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch = len(labels)
            noise = torch.randn(batch, LATENT_DIM, device=device)
            fake = generator(noise, labels)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits, real_codes = discriminator(real, labels)
            fake_logits, fake_codes = discriminator(fake.detach(), labels)
            wrong_labels = (labels + torch.randint(1, NUM_CLASSES, (batch,), device=device)) % NUM_CLASSES
            wrong_logits, _ = discriminator(real, wrong_labels)
            loss_adversarial_d = (
                criterion(real_logits, torch.ones_like(real_logits))
                + criterion(fake_logits, torch.zeros_like(fake_logits))
                + criterion(wrong_logits, torch.zeros_like(wrong_logits))
            ) / 3
            loss_q_real = code_criterion(real_codes, labels)
            loss_q_fake = code_criterion(fake_codes, labels)
            loss_d = (
                loss_adversarial_d + args.real_class_weight * loss_q_real
                + args.mi_weight * loss_q_fake
            )
            loss_d.backward()
            optimizer_d.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            generated_logits, generated_codes = discriminator(fake, labels)
            loss_adversarial_g = criterion(generated_logits, torch.ones_like(generated_logits))
            loss_info_g = code_criterion(generated_codes, labels)
            loss_g = loss_adversarial_g + args.mi_weight * loss_info_g
            loss_g.backward()
            optimizer_g.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            sum_d += loss_d.item()
            sum_g += loss_g.item()
            sum_q_real += loss_q_real.item()
            sum_q_fake += loss_q_fake.item()
            sum_match += real_logits.sigmoid().mean().item()
            sum_mismatch += wrong_logits.sigmoid().mean().item()
            correct_q_real += (real_codes.argmax(1) == labels).sum().item()
            correct_q_fake += (fake_codes.argmax(1) == labels).sum().item()
            examples += batch
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break

        print(f"epoch {epoch:>3}/{args.epochs}: D={sum_d / steps:.4f} "
              f"G={sum_g / steps:.4f} Q(real)={sum_q_real / steps:.4f} "
              f"Q(fake)={sum_q_fake / steps:.4f} "
              f"D(match)={sum_match / steps:.2f} D(wrong)={sum_mismatch / steps:.2f} "
              f"acc(real)={correct_q_real / examples:.1%} "
              f"acc(fake)={correct_q_fake / examples:.1%} steps={steps}", flush=True)
        torch.save(
            {"architecture": ARCHITECTURE, "generator": generator.state_dict(),
             "discriminator": discriminator.state_dict(), "epoch": epoch, "seed": args.seed},
            args.out / "generator.pt",
        )
        generate_grid(generator, device, args.out / f"epoch-{epoch:03d}.png", seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 16x16 MNIST cGAN")
    parser.add_argument("--data", type=Path, default=Path("data/mnist"))
    parser.add_argument("--out", type=Path, default=Path("runs/cnn-conditional-q-projection"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--mi-weight", type=float, default=1.0,
                        help="weight for Q(c|generated image) cross-entropy")
    parser.add_argument("--real-class-weight", type=float, default=1.0,
                        help="weight for Q(c|real image) supervision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="limit batches per epoch for a smoke run")
    parser.add_argument("--describe", action="store_true", help="print shapes and parameter counts")
    args = parser.parse_args()
    describe()
    if not args.describe:
        if (args.epochs < 1 or args.batch_size < 1 or args.max_steps < 0
                or args.mi_weight < 0 or args.real_class_weight < 0):
            parser.error("epochs and batch-size must be positive; steps and loss weights nonnegative")
        train(args)
