import torch
from data.load_data import load_data
from WGAN.trainer import train_WGAN
from utilities import generate_samples, save_graph
from WGAN.MNISTClasses import Critic, Generator, Critic_Insp, Generator_Insp
import os
from classes import Generator256, Discriminator256
from pytorchsummary import summary

def W_GAN_MNIST(batch_size=64, g_lr=0.0001, d_lr=0.0001, d_updates=5, z_dim=64, G=None, D=None):
    dataloader = load_data(batch_size=batch_size, MNIST=True)
    beta_1 = 0.5
    beta_2 = 0.999
    n_epochs = 100

    if not D and not G:
        D = Critic()
        G = Generator(z_dim)

    data_path = f'./results/b_{batch_size}_glr_{g_lr}_dlr_{d_lr}'
    if not os.path.exists(data_path):
        print("Directory does not exist. \nCreating directory ....", end=" ")
        os.makedirs(data_path)
        print("Created!\n")
    else:
        print("Directory already exists.\n")

    y_G, y_D = train_WGAN(G, D, dataloader, latent_dim=z_dim, batch_size=batch_size,
                          device='cuda', g_lr=g_lr, d_lr=d_lr, n_epochs=n_epochs, c_lambda=10,
                          d_updates=d_updates, learning_betas=(beta_1, beta_2),
                          output_path=f'{data_path}')

    save_graph(y_G, y_D)

    G.to('cpu')
    D.to('cpu')
    G, D = G.eval(), D.eval()

    fake_samples = generate_samples(G, f'{data_path}/final.jpg',
                                    latent_d=z_dim, num_samples=64)

    # save  models
    torch.jit.save(torch.jit.trace(G, torch.rand(batch_size, z_dim, 1, 1)),
                   f'{data_path}/G_model_MNIST.pth')
    torch.jit.save(torch.jit.trace(D, fake_samples),
                   f'{data_path}/D_model_MNIST.pth')


def W_GAN_SCULPTURES(batch_size=64, g_lr=0.0001, d_lr=0.0001, d_updates=5, z_dim=64):
    dataloader, _ = load_data(download=False, batch_size=batch_size, MNIST=False, size=(256, 256))
    beta_1 = 0.5
    beta_2 = 0.999
    n_epochs = 5000

    D = Discriminator256()
    G = Generator256()

    data_path = f'./results/sculptures/b_{batch_size}_glr_{g_lr}_dlr_{d_lr}'
    if not os.path.exists(data_path):
        print("Directory does not exist. \nCreating directory ....", end=" ")
        os.makedirs(data_path)
        print("Created!\n")
    else:
        print("Directory already exists.\n")

    y_G, y_D = train_WGAN(G, D, dataloader, latent_dim=z_dim, batch_size=batch_size,
                          device='cuda', g_lr=g_lr, d_lr=d_lr, n_epochs=n_epochs, c_lambda=10,
                          d_updates=d_updates, learning_betas=(beta_1, beta_2),
                          output_path=f'{data_path}')

    save_graph(y_G, y_D)

    G.to('cpu')
    D.to('cpu')
    G, D = G.eval(), D.eval()

    fake_samples = generate_samples(G, f'{data_path}/final.jpg',
                                    latent_d=z_dim, num_samples=8)

    # save  models
    torch.jit.save(torch.jit.trace(G, torch.rand(batch_size, z_dim, 1, 1)),
                   f'{data_path}/G_model.pth')
    torch.jit.save(torch.jit.trace(D, fake_samples),
                   f'{data_path}/D_model.pth')
