import torch
from torch import nn
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import os
from torchvision.utils import save_image

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
# BATCH_SIZE = 16


class AdaptiveAugmenter(nn.Module):
    def __init__(self, batch_size, size=256, target_accuracy=0.85, device='cpu'):
        super().__init__()
        self.device = device
        self.target_accuracy = target_accuracy
        self.batch_size = batch_size
        self.probability = torch.tensor(0.0, requires_grad=True, dtype=torch.float16)
        self.resizer = transforms.Resize(size=(size, size))
        self.resizer2 = transforms.RandomResizedCrop(size=(256, 256), scale=(0.6, 1.4), ratio=(0.8, 1.2))

    def forward(self, images):
        print('prob: ', self.probability.item())
        pipe = self.constructPipe()
        pipe = transforms.Compose(pipe)

        augmented_images = pipe(images)
        if torch.randn(1).item() <= self.probability.item():
            augmented_images = transforms.functional.rotate(augmented_images, angle=90)

        augmented_images = self.resizer2(augmented_images)
        # augmented_images = self.resizer(augmented_images)

        augmentation_values = torch.rand(self.batch_size, 1, 1, 1, device=self.device)
        augmentation_bools = augmentation_values < self.probability

        images = self.resizer2(images)
        # augmentation_bools.to(self.device)
        out_images = torch.where(augmentation_bools, augmented_images, images)

        return out_images

    def constructPipe(self):
        pipe = []

        # pipe.append(transforms.RandomResizedCrop(size=(256, 256), scale=(0.6, 1.4), ratio=(0.8, 1.2)))
        # images = transforms.functional.rotate(images, angle=90)

        if torch.randn(1).item() <= self.probability.item():
            pipe.append(transforms.RandomHorizontalFlip(p=1.0))
        if torch.randn(1).item() <= self.probability.item():
            pipe.append(transforms.RandomVerticalFlip(p=1.0))
        if torch.randn(1).item() <= self.probability.item():
            pipe.append(transforms.ColorJitter(brightness=(0.5, 1.5)))
        if torch.randn(1).item() <= self.probability.item():
            pipe.append(transforms.ColorJitter(hue=(-0.3, 0.3)))
        if torch.randn(1).item() <= self.probability.item():
            pipe.append(transforms.ColorJitter(contrast=(0.5, 1.5)))
        if torch.randn(1).item() <= self.probability.item():
            pipe.append(transforms.ColorJitter(saturation=(0.5, 1.5)))
        if torch.randn(1).item() <= self.probability.item():
            pipe.append(transforms.RandomErasing(p=1, scale=(0.02, 0.25), ratio=(0.7, 1.3)))

        return pipe

    def update(self, real_logits):
        current_accuracy = real_logits.mean()
        accuracy_error = current_accuracy - self.target_accuracy
        integration_steps = 1000
        print(f'stepping: {accuracy_error}')
        torch.clamp(self.probability + accuracy_error / integration_steps, min=0.0, max=1.0)

#
# data_transform = transforms.Compose([
#     transforms.Resize(size=(256, 256)),
#     transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
# ])
# train_data = datasets.ImageFolder(root="../data/dataset/",
#                                   transform=data_transform)
# train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
# images = next(iter(train_dataloader))
# images = images[0]
# # for index, img in enumerate(images):
# #     img_permute = img.permute(1, 2, 0)      #[height, width, color_channels]
# #     plt.figure(figsize=(10, 7))
# #     plt.imshow(img_permute)
# #     plt.axis("off")
# #     plt.title(f'Image: {index}', fontsize=14)
# #     plt.show()
# #     plt.close()
#
# ada = AdaptiveAugmenter(batch_size=BATCH_SIZE)
# for i in range(3):
#     images2 = ada(images)
#     for index, img in enumerate(images2):
#         # img_permute = img.permute(1, 2, 0)      #[height, width, color_channels]
#         # plt.figure(figsize=(10, 7))
#         # plt.imshow(img_permute)
#         # plt.axis("off")
#         # plt.title(f'Image: {index}', fontsize=14)
#         # plt.show()
#         # plt.close()
#         if not os.path.exists(f'things/epoch{i}'):
#             os.makedirs(f'things/epoch{i}')
#         save_image(img, f"things/epoch{i}/img_{index}.png")
