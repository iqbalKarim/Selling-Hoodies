import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

torch.manual_seed(0)  # Set for testing purposes, please do not change!


def plot_images_from_tensor(image_tensor, num_images=25, size=(1, 28, 28)):
    """
    Plots a grid of images from a given tensor.

    The function first scales the image tensor to the range [0, 1]. It then detaches the tensor from the computation
    graph and moves it to the CPU if it's not already there. After that, it creates a grid of images and plots the grid.

    Args:
        image_tensor (torch.Tensor): A 4D tensor containing the images.
            The tensor is expected to be in the shape (batch_size, channels, height, width).
        num_images (int, optional): The number of images to include in the grid. Default is 25.
        size (tuple, optional): The size of a single image in the form of (channels, height, width). Default is (1, 28, 28).

    Returns:
        None. The function outputs a plot of a grid of images.
    """

    # Normalize the image tensor to [0, 1]
    image_tensor = (image_tensor + 1) / 2

    # Detach the tensor from its computation graph and move it to the CPU
    img_detached = image_tensor.detach().cpu()

    # Create a grid of images using the make_grid function from torchvision.utils
    image_grid = make_grid(img_detached[:num_images], nrow=5)

    # Plot the grid of images
    # The permute() function is used to rearrange the dimensions of the grid for plotting
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def weights_init(m):
    # Check if the module 'm' is an instance of Conv2d or ConvTranspose2d
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # Initialize the weights with a normal distribution with mean 0.0 and standard deviation 0.02
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    # Check if the module 'm' is an instance of BatchNorm2d
    if isinstance(m, nn.BatchNorm2d):
        # Initialize the weights with a normal distribution with mean 0.0 and standard deviation 0.02
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        # Initialize the biases with a constant value of 0
        torch.nn.init.constant_(m.bias, 0)


def get_noise(n_samples, z_dim, device="cpu"):
    return torch.randn(n_samples, z_dim, 1, 1, device=device)


# Generator Loss Calculation


def get_gen_loss(critic_fake_prediction):
    """
    #### Generator Loss = -[average critic score on fake images]

    Generator Loss: D(G(z))

    The generator tries to maximize this function. In other words,
    It tries to maximize the discriminator's output for its fake instances. In these functions: """
    gen_loss = -1.0 * torch.mean(critic_fake_prediction)
    return gen_loss

# Critic Loss Calculation
def get_crit_loss(critic_fake_prediction, crit_real_pred, gp, c_lambda):

    """The math for the loss functions for the critic and generator is:

    Critic Loss: D(x) - D(G(z))

    Now for the Critic Loss, as per the Paper, we have to maximize the above expression.
    So, arithmetically, maximizing an expression, means minimizing the -ve of that expression

    i.e. -(D(x) - D(G(z)))
    i.e. -D(x) + D(G(z))
    i.e. -D(real_imgs) + D(G(real_imgs))
    i.e. -D(real_imgs) + D(fake_imgs)
    """
    crit_loss = (
        torch.mean(critic_fake_prediction) - torch.mean(crit_real_pred) + c_lambda * gp
    )
    return crit_loss
