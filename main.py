import torch

from load_data import load_data
from trainer import train_WGAN
from trainer2 import train_wgan
from classes import Discriminator, Generator, Discriminator256, Generator256
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from utilities import generate_samples


if __name__ == '__main__':
    # model_path = "./models/"
    # print(torch.cuda.is_available())
    # print('\ndevice name:\n', torch.cuda.get_device_name())
    # print('\ncapability:\n', torch.cuda.get_device_capability())

    # train_WGAN(dataloader)

    batch_size = 64
    dataloader = load_data(download=False, batch_size=batch_size, MNIST=False, show_samples=False, size=(256, 256))

    D = Discriminator256()
    G = Generator256()
    train_wgan(D, G, dataloader, 64, batch_size=batch_size, device='cuda')

    G.to('cpu')
    D.to('cpu')
    G, D = G.eval(), D.eval()

    fake_samples = generate_samples(G, 'final')

    # save  models
    torch.jit.save(torch.jit.trace(G, torch.rand(batch_size, 64, 1, 1)), 'results/G_model256.pth')
    torch.jit.save(torch.jit.trace(D, fake_samples), 'results/D_model256.pth')

