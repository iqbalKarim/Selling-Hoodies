import torch
from torch import optim
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from math import log2
from tqdm import tqdm
from stylegan_utils import gradient_penalty, generate_examples, save_model
from PIL import Image, ImageFile
from classes import Generator, Discriminator
import numpy

DATASET = "/vol/bitbucket/ik323/fyp/dataset"
START_TRAIN_AT_IMG_SIZE = 8 #The authors start from 8x8 images instead of 4x4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZES = [512, 256, 128, 64, 32, 16, 4]
CHANNELS_IMG = 3
Z_DIM = 256
W_DIM = 256
IN_CHANNELS= 256
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

# print(BATCH_SIZES[int(log2(512 / 8))])
print(f"Using: {DEVICE}")
def get_loader(image_size=256, device='cpu'):
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    print(f'Batch Size: {batch_size}, Image Size: {image_size}')
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    # dataset_subset = torch.utils.data.Subset(dataset, numpy.random.choice(len(dataset), 256, replace=False))
    loader = DataLoader(
        dataset,
        num_workers=6,
        batch_size=batch_size,
        shuffle=True,
    )
    return loader, dataset


def trainer(generator, critic, step, alpha, opt_critic, opt_gen, scaler_c, scaler_g, z_dim=256, device='cpu',
            lamda_gp=10):
    dataloader, dataset = get_loader(image_size=4*2**step)

    loop = tqdm(dataloader, leave=False)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, z_dim).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            fake = generator(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake, alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, device=device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            loss_critic = (- (torch.mean(critic_real) - torch.mean(critic_fake)) + lamda_gp * gp
                           + (0.001 * torch.mean(critic_real ** 2)))

        opt_critic.zero_grad()
        scaler_c.scale(loss_critic).backward(retain_graph=True)
        scaler_c.step(opt_critic)
        scaler_c.update()

        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_g.scale(loss_gen).backward(retain_graph=True)
        scaler_g.step(opt_gen)
        scaler_g.update()

        alpha += cur_batch_size / ((PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item(), loss_gen=loss_gen.item())
        # loop.set_description(f'Epoch {str(0)}')

    return alpha


def tester():
    generator = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    critic = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)

    # print(generator)
    # print(critic)

    # initialize optimizers
    opt_gen = optim.Adam([{"params": [param for name, param in generator.named_parameters() if "map" not in name]},
                          {"params": generator.map.parameters(), "lr": 1e-5}], lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99)
    )

    scaler_g = torch.amp.GradScaler()
    scaler_c = torch.amp.GradScaler()

    generator.train()
    critic.train()

    step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        print(f'Current image size: {4 * 2 ** step}')

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            alpha = trainer(generator, critic, step, alpha, opt_critic, opt_gen,
                            scaler_c, scaler_g, device=DEVICE, z_dim=Z_DIM)
        save_model(generator, critic, opt_gen, opt_critic, alpha,
                   Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG,
                   step, identifier=f'step{step}_alpha{alpha}')
        generate_examples(generator, step, z_dim=Z_DIM, n=50, device=DEVICE)
        step += 1

    # noise = torch.randn(1, z_dim).to(device)
    # img = gen(noise, alpha, steps)
    save_model(generator, critic, opt_gen, opt_critic, alpha,
               Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG,
               step, identifier='final')


tester()
