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

def generate_examples(gen, epoch, n=50):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w = get_w(1)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples2/epoch{epoch}'):
                os.makedirs(f'saved_examples2/epoch{epoch}')
            save_image(img * 0.5 + 0.5, f"saved_examples2/epoch{epoch}/img_{i}.png")

    gen.train()


def get_loader():
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    transform = transforms.Compose(
        [
            transforms.Resize((2 ** LOG_RESOLUTION, 2 ** LOG_RESOLUTION)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    dataset_subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset),
                                             ((len(dataset) // BATCH_SIZE) * BATCH_SIZE),
                                             replace=False))
    loader = DataLoader(
        dataset_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        prefetch_factor=2,
        pin_memory=True
    )
    return loader

print('Getting loader...')
loader = get_loader()

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


def get_w(batch_size):
    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)


def trainer(critic, gen, path_length_penalty, loader, opt_critic, opt_gen, opt_mapping_network, epoch):

    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        w = get_w(cur_batch_size)
        noise = get_noise(cur_batch_size)
        print('these', len(w), w.shape)
        print(w[0].shape, noise[0][1].shape)
        with torch.cuda.amp.autocast():
            fake = gen(w, noise)
            critic_fake = critic(fake.detach())
            critic_real = critic(real)

            gp = gradient_penalty(critic, real, fake, device=DEVICE)
            loss_critic = (
                - (torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake)
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
    print(f"Using {DEVICE}")
    for epoch in range(EPOCHS):
        trainer(
            critic, gen, path_length_penalty, loader, opt_critic, opt_gen, opt_mapping_network, epoch
        )
        if epoch % 10 == 0:
            generate_examples(gen, epoch)
            save_everything(critic, gen, path_length_penalty, mapping_network, opt_critic, opt_gen, opt_mapping_network, epoch)

# 'generator': gen.state_dict(),
# 'discriminator': critic.state_dict(),
# 'mapping': mapping_network.state_dict(),
# 'plp': path_length_penalty.state_dict(),
# 'g_optim': opt_gen.state_dict(),
# 'd_optim': opt_critic.state_dict(),
# 'map_optim': opt_mapping_network.state_dict()
def continue_training(identifier, curr_epoch):
    print(f"Using {DEVICE}")

    model = torch.load(f'./models2/{identifier}/trained.pth')
    critic.load_state_dict(model["discriminator"])
    gen.load_state_dict(model["generator"])
    mapping_network.load_state_dict(model["mapping"])
    path_length_penalty.load_state_dict(model["plp"])
    opt_gen.load_state_dict(model["g_optim"])
    opt_critic.load_state_dict(model["d_optim"])
    opt_mapping_network.load_state_dict(model["map_optim"])

    for epoch in range(curr_epoch + 1, EPOCHS):
        trainer(
            critic, gen, path_length_penalty, loader, opt_critic, opt_gen, opt_mapping_network, epoch
        )
        if epoch % 10 == 0:
            generate_examples(gen, epoch)
            save_everything(critic, gen, path_length_penalty, mapping_network, opt_critic, opt_gen, opt_mapping_network, epoch)

if __name__ == "__main__":
    print('training...')
    tester()
    # continue_training('epoch110', 110)
