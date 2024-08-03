from torch import optim
import torch
from tqdm import tqdm
from utils import get_noise, gradient_penalty, save_everything
import torchvision.transforms.v2 as transforms
from config import *
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from classes import Generator, Discriminator, MappingNetwork, PathLengthPenalty
import os
from torchvision.utils import save_image
import numpy as np
from math import log2


def generate_examples(generator, step, n=50):
    generator.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w = get_w(1, LOG_RESOLUTION - step + 1)
            noise = get_noise(1, LOG_RESOLUTION - step + 1)
            with torch.cuda.amp.autocast():
                img = generator(w, noise, DEVICE, step)
            if not os.path.exists(f'saved_examples/step{step}'):
                os.makedirs(f'saved_examples/step{step}')
            save_image(img * 0.5 + 0.5, f"saved_examples/step{step}/img_{i}.png")

    generator.train()


def get_loader(log_resolution, batch_size, step):
    print('Loading Dataset...', end="\t\t")
    size = min(256, 2 ** (step + 2))
    # size = 2 ** 3 + step
    print('Image size:', (size, size), end="\t")
    cur_b_size = BATCH_SIZES[int(log2(size) - 3)]
    print("Batch size:", cur_b_size)

    transform = transforms.Compose(
        [
            # transforms.Resize((2 ** log_resolution, 2 ** log_resolution)),
            transforms.Resize((size, size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )
    dataset = datasets.ImageFolder(root=DATASET, transform=transform, )
    # strict complete batch
    dataset_subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset),
                                                                       ((len(dataset) // batch_size) * batch_size),
                                                                       replace=False))
    # dataset_subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset),
    #                                                                    (12 * cur_b_size),
    #                                                                    replace=False))
    print('Prepping loader...')
    loader = DataLoader(
        dataset_subset,
        batch_size=cur_b_size,
        shuffle=True,
        num_workers=6,
        prefetch_factor=2,
        pin_memory=True
    )
    return loader, dataset


def get_w(batch_size, log_resolution):
    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(log_resolution, -1, -1)


def trainer(critic, gen, path_length_penalty, loader, opt_critic,
            opt_gen, opt_mapping_network, epoch, step):

    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        w = get_w(cur_batch_size, LOG_RESOLUTION - step + 1)
        noise = get_noise(cur_batch_size, LOG_RESOLUTION - step + 1)
        with torch.cuda.amp.autocast():
            critic_step = gen.n_blocks - step

            fake = gen(w, noise, DEVICE, step)

            critic_fake = critic(fake.detach(), critic_step)
            critic_real = critic(real, critic_step)

            gp = gradient_penalty(critic, real, fake, critic_step, device=DEVICE)
            loss_critic = (
                - (torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake, critic_step)
        loss_gen = -torch.mean(gen_fake)

        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        torch.cuda.empty_cache()

        mapping_network.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
            loss_gen=loss_gen.item(),
            gpu_mem=(torch.cuda.memory_reserved(0)/1024/1024/1024)
        )
        loop.set_description(f"Epoch {epoch}")
        torch.cuda.empty_cache()


def tester():
    # for step in range(1,2):
    for step in range(gen.n_blocks - 1, 0, -1):
        loader, _ = get_loader(step, BATCH_SIZE, gen.n_blocks - step)
        for epoch in range(STEP_EPOCHS):
            trainer(critic, gen, path_length_penalty, loader, opt_critic,
                    opt_gen, opt_mapping_network, epoch, step)
            # if epoch % 10 == 0:
        generate_examples(gen, step, n=10)
        save_everything(critic, gen, path_length_penalty, mapping_network,
                        opt_critic, opt_gen, opt_mapping_network, epoch, step)


if __name__ == "__main__":
    print(f'Using {DEVICE}')

    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print('Constructing models...')
    gen = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
    critic = Discriminator(LOG_RESOLUTION).to(DEVICE)
    mapping_network = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)
    path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    gen.train()
    critic.train()
    mapping_network.train()
    print('Training...')
    tester()
