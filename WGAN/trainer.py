import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from WGAN.wgan_utils import *
from utilities import generate_fake_images_from_generator, generate_samples
from torch.cuda.amp import GradScaler


def gradient_of_critic_score(critic, real, fake, epsilon):
    """
    Function to compute the gradient of the critic's scores for interpolated images.

    This function is a key component of the Gradient Penalty in WGAN-GP (Wasserstein GAN with Gradient Penalty),
    a popular GAN architecture. The gradient penalty encourages the critic's gradient norms to be close to 1,
    which ensures the 1-Lipschitz constraint needed for the Wasserstein distance function to be valid.

    Args:
        critic (nn.Module): The critic model, typically a neural network.
        real (torch.Tensor): Batch of real images.
        fake (torch.Tensor): Batch of fake images generated by the generator.
        epsilon (float): The weight for the interpolation between real and fake images.

    Returns:
        gradient (torch.Tensor): The computed gradient of the critic's scores for the interpolated images.
    """

    # Create the interpolated images as a weighted combination of real and fake images
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty_l2_norm(gradient):
    """
    Calculate the L2 norm of the gradient for enforcing the 1-Lipschitz constraint in Wasserstein GAN with Gradient Penalty (WGAN-GP).

    The gradient penalty is calculated as the mean square error of the gradient norms from 1. The gradient penalty encourages the gradients of the critic to be unit norm, which is a key property of 1-Lipschitz functions.

    Args:
    gradient (torch.Tensor): The gradients of the critic's scores with respect to the interpolated images.

    Returns:
    torch.Tensor: The gradient penalty.
    """
    # Reshape each image in the batch into a 1D tensor (flatten the images)
    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)

    # Calculate the penalty as the mean squared distance of the norms from 1.
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty

def train_WGAN(generator, discriminator, dataloader, batch_size,
               g_lr=0.0001, d_lr=0.0001, learning_betas=(0.0, 0.9),
               c_lambda=10, display_step=500,
               d_updates=1, n_epochs=20, latent_dim=64, device='cpu',
               output_path='./results'):

    generator = generator.apply(weights_init)
    discriminator = discriminator.apply(weights_init)
    gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=g_lr, betas=learning_betas)
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=d_lr, betas=learning_betas)

    generator.to(device)
    discriminator.to(device)

    scalerG = GradScaler()
    scalerD = GradScaler()

    current_step = 0
    generator_losses = []
    discriminator_losses = []
    critic_losses_across_critic_repeats = []

    for epoch in range(n_epochs):
        loop = tqdm(dataloader, leave=False)
        for batch_idx, (real, _) in enumerate(loop):
            cur_batch_size = len(real)
            real = real.to(device)

            mean_critic_loss_for_this_iteration = 0
            for _ in range(d_updates):
                #  Train Critic
                disc_optimizer.zero_grad()
                fake_noise = get_noise(cur_batch_size, latent_dim, device=device)

                with torch.autocast(device_type=device, dtype=torch.float16):
                    fake = generator(fake_noise)
                    critic_fake_prediction = discriminator(fake.detach())
                    crit_real_pred = discriminator(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                # epsilon will be a Tensor of size torch.Size([128, 1, 1, 1]) for batch_size of 128
                gradient = gradient_of_critic_score(discriminator, real, fake.detach(), epsilon)
                gp = gradient_penalty_l2_norm(gradient)

                with torch.autocast(device_type=device, dtype=torch.float16):
                    crit_loss = get_crit_loss(critic_fake_prediction, crit_real_pred, gp, c_lambda)

                # Keep track of the average critic loss in this batch
                mean_critic_loss_for_this_iteration += crit_loss.item() / d_updates

                # Update gradients
                # crit_loss.backward(retain_graph=True)
                # Update the weights
                # disc_optimizer.step()
                scalerD.scale(crit_loss).backward(retain_graph=True)
                scalerD.step(disc_optimizer)
                scalerD.update()

                discriminator_losses += [crit_loss.item()]

            critic_losses_across_critic_repeats += [mean_critic_loss_for_this_iteration]

            #  Train Generators
            gen_optimizer.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, latent_dim, device=device)

            with torch.autocast(device_type=device, dtype=torch.float16):
                fake_2 = generator(fake_noise_2)
                critic_fake_prediction = discriminator(fake_2)
                # Update the gradients
                gen_loss = get_gen_loss(critic_fake_prediction)


            # Update the weights
            # gen_loss.backward()
            # gen_optimizer.step()
            scalerG.scale(gen_loss).backward()
            scalerG.step(gen_optimizer)
            scalerG.update()

            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss_critic=crit_loss.item(), loss_gen=gen_loss.item())

        if epoch % 10 == 0:
            generate_fake_images_from_generator(generator, latent_dim, batch_size,
                                                file_path=f"{output_path}/{epoch}.png", device="cuda")
            fake_samples = generate_samples(generator, f'{output_path}/checkpoint_{epoch}.jpg',
                                            latent_d=latent_dim, num_samples=8, device=device)
            torch.jit.save(torch.jit.trace(generator, torch.rand(batch_size, latent_dim, 1, 1, device=device)),
                           f'{output_path}/{epoch}_G_model.pth')
            torch.jit.save(torch.jit.trace(discriminator, fake_samples),
                           f'{output_path}/{epoch}_D_model.pth')

        generator_mean_loss_display_step = sum(generator_losses[-display_step:]) / display_step
        critic_mean_loss_display_step = sum(critic_losses_across_critic_repeats[-display_step:]) / display_step
        print(
            f"Step {current_step}: Generator loss: {generator_mean_loss_display_step}, "
            f"critic loss: {critic_mean_loss_display_step}"
        )

        step_bins = 20
        num_examples = (len(generator_losses) // step_bins) * step_bins

        plt.plot(
            range(num_examples // step_bins),
            torch.Tensor(generator_losses[:num_examples])
            .view(-1, step_bins)
            .mean(1),
            label="Generator Loss",
        )
        plt.plot(
            range(num_examples // step_bins),
            torch.Tensor(critic_losses_across_critic_repeats[:num_examples])
            .view(-1, step_bins)
            .mean(1),
            label="Critic Loss",
        )
        plt.legend()
        plt.savefig(f"{output_path}/training_graph.jpg")
        plt.close()

    return discriminator_losses, generator_losses
