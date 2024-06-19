import torch

from load_data import load_data
from trainer import train_WGAN
from trainer2 import train_wgan
from classes import Discriminator, Generator
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_path = "./models/"

    # with open("./data/omniart_v3_datadump.csv") as input_file:
    #     head = [next(input_file) for _ in range(1)]
    # print(head)

    # df = pd.read_csv("./data/omniart_v3_datadump.csv")
    # image_urls = df["image_url"].tolist()
    # # image_urls.to_csv("./data/image_urls.csv", index=False)
    # with open("./data/image_urls.csv", 'w', newline='') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(image_urls)

    batch_size = 64
    dataloader = load_data(download=False, batch_size=batch_size, MNIST=False)
    # train_WGAN(dataloader)

    D = Discriminator()
    G = Generator()
    train_wgan(D, G, dataloader, 64)

    G, D = G.eval(), D.eval()

    with torch.no_grad():
        noise = torch.rand(batch_size, 64, device="cpu")
        fake_samples_w = G(noise)
        generated = make_grid(fake_samples_w, nrow=10, padding=2, normalize=False,
                              value_range=None, scale_each=False, pad_value=0)
        save_image(fake_samples_w, './final.png')

    # save  models
    torch.jit.save(torch.jit.trace(G, noise), 'W_GAN/G_model.pth')
    torch.jit.save(torch.jit.trace(D, fake_samples_w), 'W_GAN/D_model.pth')