import torch
import os
from torchvision.utils import save_image
from stylegan_utils import load_all
from classes import Generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS_IMG = 3
Z_DIM = 256
W_DIM = 256
IN_CHANNELS = 256


def generate_examples(gen, steps, z_dim, n=100, device='cpu', uniq_path="saved_examples"):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, z_dim).to(device)
            img = gen(noise, alpha, steps)
            path = f'{uniq_path}/step{steps}'
            if not os.path.exists(path):
                os.makedirs(path)
            save_image(img, f"{path}/img_{i}.png")
    gen.train()


if __name__ == "__main__":
    generator = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    saved = load_all("step4_alpha1")
    generator.load_state_dict(saved['generator'])

    generate_examples(generator, 4, Z_DIM, n=50, device=DEVICE, uniq_path="testing")

