import torch.optim
from classes import Discriminator, Generator
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np

def train_wgan(D, G, loader, latent_d, batch_size, epochs=20, d_updates=1, device="cpu"):
    G_losses = []
    D_losses = []
    scaler = GradScaler()


    # Send to device which available
    G.to(device)
    D.to(device)

    # Adam optimizers
    optimizerD = torch.optim.AdamW(D.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerG = torch.optim.AdamW(G.parameters(), lr=0.0001, betas=(0.0, 0.9))

    for epoch in tqdm(range(epochs)):
        print('Epoch: ', epoch)
        for count, data in enumerate(tqdm(loader, leave=False)):
            if isinstance(data, tuple) or len(data) == 2:
                data, class_label = data
            elif isinstance(data, list) and len(data) == 1:
                data = data[0]

            batch_size_internal = data.size(0)
            real = data.to(device)

            # print(real.size())

            D.zero_grad()
            G.zero_grad()

            # Step 1) D-Score, G-Score, Gradient penalty

            # How well does D work on real data
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                D_success = D(real)

                # train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(batch_size_internal, latent_d, 1, 1, device=device)
                # Generate fake images with G
                fake = G(noise)
                del noise
                # Classify all fake batch with D
                D_failure = D(fake)

                # Gradient Penalty
                eps_shape = [batch_size_internal] + [1] * (len(data.shape)-1)
                eps = torch.rand(eps_shape, device=device)
                fake = eps*real + (1-eps) * fake
                output = D(fake)

                grad = torch.autograd.grad(outputs=output, inputs=fake,
                                           grad_outputs=torch.ones(output.size(), device=device),
                                           create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]

                del fake, eps, eps_shape, output

                D_grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()

                # Calculate D's loss on all fake batch
                err_D = (D_failure - D_success).mean() + D_grad_penalty.mean()*10
                del D_failure, D_success, D_grad_penalty

            scaler.scale(err_D).backward()
            # err_D.backward()
            scaler.step(optimizerD)
            # optimizerD.step()
            scaler.update()
            D_losses.append(err_D.item())

            if count % d_updates != d_updates-1:
                continue

                # Step 2) -D(G(z))
            D.zero_grad()
            G.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                noise = torch.randn(batch_size_internal, latent_d, 1, 1, device=device)
                output = -D(G(noise))
                del noise
                err_G = output.mean()

            scaler.scale(err_G).backward()
            # err_G.backward()
            scaler.step(optimizerG)
            # optimizerG.step()
            scaler.update()

            torch.cuda.empty_cache()
            G_losses.append(err_G.item())

        with torch.no_grad():
            noise = torch.rand(batch_size, 64, 1, 1, device="cuda")
            generated = G(noise)
            generated = make_grid(generated, nrow=2, padding=2, normalize=False,
                                  value_range=None, scale_each=False, pad_value=0)
            generated = generated.cpu()
            npimg = generated.numpy()
            plt.figure(figsize=(15, 15))
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            save_image(generated, f'./results/{epoch}.png')

    return G_losses, D_losses
