import torch
from torch import optim
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from math import log2
from tqdm import tqdm
from stylegan_utils import gradient_penalty, generate_examples, save_model, load_model, load_all
from PIL import Image, ImageFile
from classes import Generator, Discriminator
import numpy
from augmentor import AdaptiveAugmenter

DATASET = "/vol/bitbucket/ik323/fyp/anime"
# DATASET = "../data/dataset/"
START_TRAIN_AT_IMG_SIZE = 8 #The authors start from 8x8 images instead of 4x4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
# BATCH_SIZES = [512, 64, 64, 64, 32, 16, 4]
BATCH_SIZES = [512, 256, 128, 64, 32, 16, 4]
# BATCH_SIZES = [512, 16, 8, 4, 4, 16, 4]
CHANNELS_IMG = 3
Z_DIM = 256
W_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [40] * len(BATCH_SIZES)
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

def get_loader(image_size=256, device='cpu'):
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            )
        ]
    )
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    # take subset of data such that batches are always whole.
    dataset_subset = torch.utils.data.Subset(dataset, numpy.random.choice(len(dataset),
                                             ((len(dataset)//batch_size) * batch_size), replace=False))

    loader = DataLoader(
        dataset_subset,
        num_workers=6,
        batch_size=batch_size,
        shuffle=True,
        prefetch_factor=2,
    )
    return loader, dataset


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

    # initialize optimizers
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
            alpha = trainer(generator, critic, step, alpha, opt_critic, opt_gen, device=DEVICE, z_dim=Z_DIM)
        save_model(generator, critic, opt_gen, opt_critic, alpha,
                   Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG,
                   step, identifier=f'anime_step{step}_alpha{alpha}')
        generate_examples(generator, step, z_dim=Z_DIM, n=50, device=DEVICE, uniq_path="anime")
        step += 1

    save_model(generator, critic, opt_gen, opt_critic, alpha,
               Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG,
               step, identifier='final')


def continueTraining(identifier):
    generator = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    critic = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)

    # initialize optimizers
    opt_gen = optim.Adam([{"params": [param for name, param in generator.named_parameters() if "map" not in name]},
                          {"params": generator.map.parameters(), "lr": 1e-5}], lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99)
    )

    step, alpha, _ = load_model(generator, identifier, with_critic=True, crit=critic,
                                with_optim=True, opt_gen=opt_gen, opt_crit=opt_critic)
    step = step + 1
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        print(f'Current image size: {4 * 2 ** step}')
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            alpha = trainer(generator, critic, step, alpha, opt_critic, opt_gen, device=DEVICE, z_dim=Z_DIM)
        save_model(generator, critic, opt_gen, opt_critic, alpha,
                   Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG,
                   step, identifier=f'anime_step{step}_alpha{alpha}')
        generate_examples(generator, step, z_dim=Z_DIM, n=50, device=DEVICE, uniq_path="anime")
        step += 1
    save_model(generator, critic, opt_gen, opt_critic, alpha,
               Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG,
               step, identifier='final')


if __name__ == '__main__':
    tester()
    # continueTraining('step4_alpha1')

