import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

# JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
#             119832  gpgpuALL interact    ik323  R       1:22      1 cloud-vm-47-218

def generate_samples(generator, output_name, show_seperate_plots = False, num_samples=10, latent_d=64):
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_d, 1, 1)
        samples = generator(noise)

        grid = make_grid(samples, 5)
        save_image(grid, f'results/{output_name}.jpg')

        if show_seperate_plots:
            for img in samples:
                img_permute = img.permute(1, 2, 0)    #[height, width, color_channels]
                plt.figure(figsize=(10, 7))
                plt.imshow(img_permute)
                plt.axis("off")
                plt.title('Image', fontsize=14)
                plt.show()

        return samples