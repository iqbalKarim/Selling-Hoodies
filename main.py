from WGAN.tester import W_GAN_MNIST, W_GAN_SCULPTURES

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt

def paramsearch_and_training_MNIST():
    # for batch in [128]:
    for batch in [64, 128]:
        for g_lr, d_lr in [(0.0001, 0.0001), (0.0002, 0.0001), (0.0001, 0.0002), (0.0002, 0.0002)]:
            print(f"Training for: \n "
                  f"\tBatch Size: {batch} \n"
                  f"\tGen LR: {g_lr}\n"
                  f"\tDisc LR: {d_lr}") 
            W_GAN_MNIST(batch, g_lr, d_lr)

def paramsearch_and_training_sculptures():
    for batch in [64]:
        for g_lr, d_lr in [(0.0001, 0.0001), (0.0002, 0.0001), (0.0001, 0.0002), (0.0002, 0.0002)]:
            print(f"Training for: \n "
                  f"\tBatch Size: {batch} \n"
                  f"\tGen LR: {g_lr}\n"
                  f"\tDisc LR: {d_lr}")
            W_GAN_SCULPTURES(batch, g_lr, d_lr, z_dim=150)


if __name__ == '__main__':
    # model_path = "./models/"
    # print(torch.cuda.is_available())
    # print('\ndevice name:\n', torch.cuda.get_device_name())
    # print('\ncapability:\n', torch.cuda.get_device_capability())

    # paramsearch_and_training_MNIST()
    # paramsearch_and_training_sculptures()


    data_transform = transforms.Compose([
        # Resize the images to 512x512 (default) or size tuple
        transforms.Resize(size=(256, 256)),
        # Turn the image into a torch.Tensor
        transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    ])

    train_data = datasets.ImageFolder(root="./data/dataset",
                                      transform=data_transform)
    print(f"\nTraining dataset: \n{train_data}")

    # images = train_data[0][0]
    for i in range(5):
        img = train_data[i][0]  # [color_channels, height, width]
        img_permute = img.permute(1, 2, 0)  # [height, width, color_channels]
        plt.figure(figsize=(10, 7))
        plt.imshow(img_permute)
        plt.axis("off")
        plt.title(f'Image: {i}', fontsize=14)
        plt.show()

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=64,  # how many samples per batch?
                                  num_workers=6,  # how many subprocesses to use for data loading? (higher = more)
                                  shuffle=True)
    print(f"Return training dataloader: \n\t{train_dataloader}")