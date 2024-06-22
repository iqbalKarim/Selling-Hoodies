import torch
from data.load_data import load_data
from WGAN.trainer import train_WGAN
from utilities import generate_samples, save_graph
from WGAN.MNISTClasses import Critic, Generator

def W_GAN_MNIST():
    z_dim = 100
    batch_size = 64
    dataloader = load_data(batch_size=batch_size, MNIST=True)
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    n_epochs = 50
    # D = DiscriminatorMNIST()
    # G = GeneratorMNIST()

    D = Critic()
    G = Generator(z_dim)

    y_G, y_D = train_WGAN(G, D, dataloader, latent_dim=z_dim, batch_size=batch_size,
                          device='cuda', g_lr=lr, d_lr=lr, n_epochs=n_epochs, c_lambda=10,
                          d_updates=5, learning_betas=(beta_1, beta_2))

    save_graph(y_G, y_D)

    G.to('cpu')
    D.to('cpu')
    G, D = G.eval(), D.eval()

    fake_samples = generate_samples(G, 'final', latent_d=z_dim, num_samples=64)

    # save  models
    torch.jit.save(torch.jit.trace(G, torch.rand(batch_size, z_dim, 1, 1)), 'results/G_model_MNIST.pth')
    torch.jit.save(torch.jit.trace(D, fake_samples), 'results/D_model_MNIST.pth')
