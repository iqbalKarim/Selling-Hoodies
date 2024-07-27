import torch
from torch import optim
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from math import log2
from tqdm import tqdm
from stylegan_utils import gradient_penalty, generate_examples, save_model
from PIL import Image, ImageFile
from classes import Generator, Discriminator
import matplotlib.pyplot as plt
import torchvision

DATASET = "/vol/bitbucket/ik323/fyp/mnist/"
# DATASET = "../data/mnist/"
START_TRAIN_AT_IMG_SIZE = 8 #The authors start from 8x8 images instead of 4x4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
# BATCH_SIZES = [512, 64, 64, 64]
BATCH_SIZES = [512, 256, 128, 64]
# BATCH_SIZES = [512, 16, 8, 4]
CHANNELS_IMG = 1
Z_DIM = 256
W_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

def get_loader(image_size=32, device='cpu'):
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            )
        ]
    )
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    # print(f'Batch Size: {batch_size}, Image Size: {image_size}')

    dataset = torchvision.datasets.MNIST(DATASET, train=True, transform=transform, download=True)
    # print(len(data))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6), dataset

def print_examples(loader):
    images = next(iter(loader))
    images = images[0]
    for index, img in enumerate(images[:5]):
        img_permute = img.permute(1, 2, 0)  # [height, width, color_channels]
        plt.figure(figsize=(10, 7))
        plt.imshow(img_permute)
        plt.axis("off")
        plt.title(f'Image: {index}', fontsize=14)
        plt.show()
        plt.close()

def trainer(generator, critic, step, alpha, opt_critic, opt_gen, z_dim=256, device='cpu',
            lamda_gp=10):
    dataloader, dataset = get_loader(image_size=4*2**step)

    loop = tqdm(dataloader, leave=True, unit="batch")
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)

        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, z_dim).to(device)

        fake = generator(noise, alpha, step)

        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake, alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, device=device)

        loss_critic = (- (torch.mean(critic_real) - torch.mean(critic_fake)) + lamda_gp * gp
                       + (0.001 * torch.mean(critic_real ** 2)))
        opt_critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        opt_gen.step()

        alpha += cur_batch_size / ((PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)
        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item(), loss_gen=loss_gen.item())

    return alpha


def tester():
    generator = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    critic = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)

    opt_gen = optim.Adam([{"params": [param for name, param in generator.named_parameters() if "map" not in name]},
                          {"params": generator.map.parameters(), "lr": 1e-5}], lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99)
    )

    generator.train()
    critic.train()
    step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        print(f'Current image size: {4 * 2 ** step}')
        print(f'Current Batch size: {BATCH_SIZES[int(log2((4*2**step) / 4))]}')
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            alpha = trainer(generator, critic, step, alpha, opt_critic, opt_gen,
                            device=DEVICE, z_dim=Z_DIM)
        save_model(generator, critic, opt_gen, opt_critic, alpha,
                   Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG,
                   step, identifier=f'MNIST_step{step}_alpha{alpha}')
        generate_examples(generator, step, z_dim=Z_DIM, n=50, device=DEVICE, uniq_path="MNIST")
        step += 1

    save_model(generator, critic, opt_gen, opt_critic, alpha,
               Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG,
               step, identifier='final')


if __name__ == "__main__":
    tester()


