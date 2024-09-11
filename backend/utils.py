import torch
from StyleGAN.classes import Generator
from torchvision.utils import make_grid
from stgan2helper import load_pickle, generate_images

CHANNELS_IMG = 3
Z_DIM = 256
IN_CHANNELS = 256
W_DIM = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_examples(gen, steps, n=5, normalize="True"):
    alpha = 1.0
    imgs = []
    # for i in range(n):
    #     with torch.no_grad():
    #         noise = torch.randn(1, Z_DIM)
    #         imgs.append(gen(noise, alpha, steps))
    with torch.no_grad():
        noise = torch.randn(n, Z_DIM)
        imgs = gen(noise.to(device), alpha, steps)
        if normalize == "True":
            print('normalize')
            imgs = (imgs * 0.5) + 0.5
    return imgs


def generate_grid(gen, step, cols=3, n=3):
    gen.eval()
    imgs = []
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(cols * cols, Z_DIM)
            samples = gen(noise.to(device), 1, step)
            grid = make_grid((samples*0.5) + 0.5, cols)
        imgs.append(grid)

    return imgs


def load_generator():
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG)
    model = torch.load("./models/trained.pth")
    gen.load_state_dict(model["generator"])
    gen.eval()
    return gen

def load_emnist_generator():
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG)
    model = torch.load("./models/emnist.pth")
    gen.load_state_dict(model["generator"])
    gen.eval()
    return gen

def load_mnist_generator():
    channels_img = 1
    z_dim = 256
    w_dim = 256
    in_channels = 256
    gen = Generator(z_dim, w_dim, in_channels, img_channels=channels_img)
    model = torch.load("./models/mnist.pth")
    gen.load_state_dict(model["generator"])
    gen.eval()
    return gen

def load_graffiti_generator():
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG)
    model = torch.load("./models/graffiti.pth")
    gen.load_state_dict(model["generator"])
    gen.eval()
    return gen

def generate_styleGAN1_examples(gen):
    return 'Gen these'

def load_met():
    return load_pickle()

def generate_met(G, n = 5):
    generate_images(G, outdir='out', truncation_psi=0.7, n=int(n))

