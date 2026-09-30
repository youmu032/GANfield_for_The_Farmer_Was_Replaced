"""Small convolutional conditional GAN."""

import torch
from torch import nn


LATENT_DIM = 32
LABEL_DIM = 8
IMAGE_SIZE = 16
NUM_CLASSES = 10
ARCHITECTURE = "cnn-conditional-q-projection-v1"


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, LABEL_DIM)
        self.project = nn.Sequential(
            nn.Linear(LATENT_DIM + LABEL_DIM, 16 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(8, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        conditioned = torch.cat((noise, self.label_embedding(labels)), dim=1)
        features = self.project(conditioned).reshape(-1, 16, 4, 4)
        return self.upsample(features)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, 64)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.unconditional_head = nn.Linear(64 * 4 * 4, 1)
        self.condition_projection = nn.Linear(64 * 4 * 4, 64, bias=False)
        self.code_head = nn.Linear(64 * 4 * 4, NUM_CLASSES)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The Q head only sees image features; the supplied label reaches the real/fake head.
        image_features = self.features(images).flatten(1)
        image_code = self.condition_projection(image_features)
        condition_score = (image_code * self.label_embedding(labels)).sum(dim=1) / 8
        real_fake_score = self.unconditional_head(image_features).squeeze(1) + condition_score
        return real_fake_score, self.code_head(image_features)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
