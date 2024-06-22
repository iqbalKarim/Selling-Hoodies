import torch.optim
from classes import Discriminator, Generator
import tqdm
from torch.cuda.amp import GradScaler
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from utilities import save_graph

def train_wgan(D, G, loader, latent_d, batch_size, epochs=20, d_updates=1, device="cpu", G_lr=0.0001, D_lr=0.0001):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    G_losses = []
    D_losses = []
    scaler = GradScaler()

    # Send to device which available
    G.to(device)
    D.to(device)

    # Adam optimizers
    optimizerD = torch.optim.AdamW(D.parameters(), lr=D_lr, betas=(0.0, 0.9))
    optimizerG = torch.optim.AdamW(G.parameters(), lr=G_lr, betas=(0.0, 0.9))

    # for epoch in tqdm(range(epochs)):
    #     print('Epoch: ', epoch)
    #     for count, data in enumerate(tqdm(loader, leave=False)):
    for epoch in range(epochs):
        # G_loss = 0
        # D_loss = 0
        with tqdm.tqdm(loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
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
                    print(noise.size())
                    return
                    fake = G(noise)

                    # Classify all fake batch with D
                    D_failure = D(fake)

                    # Gradient Penalty
                    eps_shape = [batch_size_internal] + [1] * (len(data.shape)-1)
                    eps = torch.rand(eps_shape, device=device)
                    fake = eps*real + (1-eps) * fake
                    output = D(fake)

                    grad = torch.autograd.grad(outputs=output, inputs=fake,
                                               grad_outputs=torch.ones(output.size(), device=device),
                                               create_graph=True, retain_graph=True, only_inputs=True, 
                                               allow_unused=True)[0]


                    D_grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()

                    # Calculate D's loss on all fake batch
                    err_D = (D_failure - D_success).mean() + D_grad_penalty.mean()*10

                scaler.scale(err_D).backward()
                # err_D.backward()
                scaler.step(optimizerD)
                # optimizerD.step()
                scaler.update()
                # D_loss += err_D.item()
                D_losses.append(err_D.item())

                if i % d_updates != d_updates-1:
                    continue

                # Step 2) -D(G(z))
                D.zero_grad()
                G.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    noise = torch.randn(batch_size_internal, latent_d, 1, 1, device=device)
                    output = -D(G(noise))
                    err_G = output.mean()

                scaler.scale(err_G).backward()
                # err_G.backward()
                scaler.step(optimizerG)
                # optimizerG.step()
                scaler.update()

                torch.cuda.empty_cache()
                # G_loss += err_G.item()
                G_losses.append(err_G.item())
                
                if i % epochs == 0:
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(Loss_D=err_D.item(), Loss_G=err_G.item())
            
        # G_losses.append(G_loss)
        # D_losses.append(D_loss)
        save_graph(G_losses, D_losses)

        with torch.no_grad():
            noise = torch.rand(batch_size, latent_d, 1, 1, device="cuda")
            generated = G(noise)
            generated = make_grid(generated, nrow=8)
            generated = generated.cpu()
            npimg = generated.numpy()
            save_image(generated, f'./results/{epoch}.png')
            plt.figure(figsize=(15, 15))
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    return G_losses, D_losses
