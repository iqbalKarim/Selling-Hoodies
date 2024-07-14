import torch
from torch import nn
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import os
from torchvision.utils import save_image
import numpy as np

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
# BATCH_SIZE = 16


class AdaptiveAugmenter(nn.Module):
    def __init__(self, batch_size, size=256, channels_img=3, target_accuracy=0.80, device='cpu', p=0.0):
        super().__init__()
        self.device = device
        self.target_accuracy = target_accuracy
        self.batch_size = batch_size
        # self.probability = torch.tensor(0.0, requires_grad=True, dtype=torch.float16)
        self.probability = p
        self.resizer = transforms.Resize(size=(size, size))
        self.resizer2 = transforms.RandomResizedCrop(size=(size, size), scale=(0.6, 1.4), ratio=(0.8, 1.2))
        self.normalizer = transforms.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            )
        self.rng = np.random.default_rng()


    def forward(self, images):
        if self.probability > 0.0:
            pipe = self.constructPipe()
            if len(pipe) > 0:
                pipe = transforms.Compose(pipe)
                augmented_images = pipe(images)
                if self.rng.random() <= self.probability:
                    augmented_images = transforms.functional.rotate(augmented_images, angle=90)

                augmented_images = self.resizer2(augmented_images)
                # augmented_images = self.resizer(augmented_images)

                augmentation_values = torch.rand(self.batch_size, 1, 1, 1, device=self.device)
                augmentation_bools = augmentation_values < self.probability

                images = self.resizer2(images)
                # augmentation_bools.to(self.device)
                out_images = torch.where(augmentation_bools, augmented_images, images)
                return self.normalizer(out_images)
            else:
                return self.normalizer(images)
        else:
            return self.normalizer(images)

    def constructPipe(self):
        pipe = []

        # pipe.append(transforms.RandomResizedCrop(size=(256, 256), scale=(0.6, 1.4), ratio=(0.8, 1.2)))
        # images = transforms.functional.rotate(images, angle=90)

        if self.rng.random() <= self.probability:
            pipe.append(transforms.RandomHorizontalFlip(p=1.0))
        if self.rng.random() <= self.probability:
            pipe.append(transforms.RandomVerticalFlip(p=1.0))
        if self.rng.random() <= self.probability:
            pipe.append(transforms.ColorJitter(brightness=(0.5, 1.5)))
        if self.rng.random() <= self.probability:
            pipe.append(transforms.ColorJitter(hue=(-0.3, 0.3)))
        if self.rng.random() <= self.probability:
            pipe.append(transforms.ColorJitter(contrast=(0.5, 1.5)))
        if self.rng.random() <= self.probability:
            pipe.append(transforms.ColorJitter(saturation=(0.5, 1.5)))
        if self.rng.random() <= self.probability:
            pipe.append(transforms.RandomErasing(p=1, scale=(0.02, 0.25), ratio=(0.7, 1.3)))

        return pipe

    def update(self, real_logits):
        current_accuracy = real_logits.mean()
        accuracy_error = current_accuracy - self.target_accuracy
        integration_steps = 1000
        self.probability = torch.clamp(self.probability + (accuracy_error * self.batch_size / integration_steps), min=0.0, max=1.0).item()
