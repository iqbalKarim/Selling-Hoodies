import torch.optim
from classes import Discriminator, Generator
from tqdm import tqdm

def train_wgan(D, G, loader, latent_d, epochs=20, d_updates=1, device="cpu"):
    G_losses = []
    D_losses = []

    # Send to device which available
    G.to(device)
    D.to(device)

    # Adam optimizers
    optimizerD = torch.optim.AdamW(D.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerG = torch.optim.AdamW(G.parameters(), lr=0.0001, betas=(0.0, 0.9))

    for epoch in tqdm(range(epochs)):
        for count, data in enumerate(tqdm(loader, leave=False)):
            if isinstance(data, tuple) or len(data) == 2:
                data, class_label = data
            elif isinstance(data, list) and len(data) == 1:
                data = data[0]

            batch_size = data.size(0)
            real = data.to(device)

            # print(real.size())

            D.zero_grad()
            G.zero_grad()

            # Step 1) D-Score, G-Score, Gradient penalty

            # How well does D work on real data
            D_success = D(real)

            # train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, latent_d, 1, 1, device=device)
            # Generate fake images with D
            fake = G(noise)
            # Classify all fake batch with D
            D_failure = D(fake)

            # Gradient Penalty
            eps_shape = [batch_size] + [1] * (len(data.shape)-1)
            eps = torch.rand(eps_shape, device=device)
            fake = eps*real + (1-eps) * fake
            output = D(fake)

            grad = torch.autograd.grad(outputs=output, inputs=fake,
                                       grad_outputs=torch.ones(output.size(), device=device),
                                       create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]

            D_grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()

            # Calculate D's loss on all fake batch
            err_D = (D_failure - D_success).mean() + D_grad_penalty.mean()*10
            err_D.backward()
            optimizerD.step()
            D_losses.append(err_D.item())

            if count % d_updates != d_updates-1:
                continue

            # Step 2) -D(G(z))
            D.zero_grad()
            G.zero_grad()

            noise = torch.randn(batch_size, latent_d, 1, 1, device=device)
            output = -D(G(noise))
            err_G = output.mean()
            err_G.backward()
            optimizerG.step()

            G_losses.append(err_G.item())

    return G_losses, D_losses