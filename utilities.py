import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid


def generate_samples(generator, output_path='results/output.jpg',
                     show_separate_plots=False, num_samples=10, latent_d=64, device="cpu"):
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_d, 1, 1, device=device)
        samples = generator(noise)
        grid = make_grid(samples, 5)
        save_image(grid, output_path)

        if show_separate_plots:
            for img in samples:
                img_permute = img.permute(1, 2, 0)    #[height, width, color_channels]
                plt.figure(figsize=(10, 7))
                plt.imshow(img_permute)
                plt.axis("off")
                plt.title('Image', fontsize=14)
                plt.show()

        return samples


def save_graph(y_D, y_G, filepath="./results"):
    # x = np.linspace(0, 10, 100)
    fig, ax = plt.subplots()
    
    ax.plot(y_D, linewidth=2.0, color='green', label='Discriminator Loss')
    ax.plot(y_G, linewidth=2.0, color='red', label='Generator Loss')
    ax.legend()
    # ax.set(xlim=(0, 50))
    ax.axis('auto')
    # plt.show()
    plt.savefig(filepath+"/training_graph.png")
    plt.close()


def generate_fake_images_from_generator(generator, latent_dim, batch_size,
                                        file_path='./results/output.jpg', device='cpu'):
    with torch.no_grad():
        noise = torch.rand(batch_size, latent_dim, 1, 1, device=device)
        generated = generator(noise)
        generated = make_grid(generated, nrow=8)
        generated = generated.cpu()
        save_image(generated, file_path)
