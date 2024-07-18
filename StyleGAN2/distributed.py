import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image, ImageFile
from tqdm import tqdm
from utils import get_noise, gradient_penalty, save_everything
import torchvision.transforms.v2 as transforms
from config import *
from torchvision import datasets
from torch.utils.data import DataLoader
from classes import Generator, Discriminator, MappingNetwork, PathLengthPenalty
import os
from torchvision.utils import save_image
import numpy as np


def generate_examples(gen, epoch, mapping_network, device, n=50):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w = get_w(1, mapping_network, device)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples2/epoch{epoch}'):
                os.makedirs(f'saved_examples2/epoch{epoch}')
            save_image(img * 0.5 + 0.5, f"saved_examples2/epoch{epoch}/img_{i}.png")

    gen.train()


def get_loader():
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    transform = transforms.Compose(
        [
            transforms.Resize((2 ** LOG_RESOLUTION, 2 ** LOG_RESOLUTION)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    dataset_subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset),
                                                                       ((len(dataset) // BATCH_SIZE) * BATCH_SIZE),
                                                                       replace=False))
    loader = DataLoader(
        dataset_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        prefetch_factor=2,
        pin_memory=True
    )
    return loader


def get_w(batch_size, mapping_network, device):
    z = torch.randn(batch_size, W_DIM).to(device)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def trainer(critic, gen, path_length_penalty, mapping_network, loader,
            opt_critic, opt_gen, opt_mapping_network, epoch, device):

    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        w = get_w(cur_batch_size, mapping_network, device)
        noise = get_noise(cur_batch_size)
        with torch.cuda.amp.autocast():
            fake = gen(w, noise)
            critic_fake = critic(fake.detach())
            critic_real = critic(real)

            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                - (torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)

        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        mapping_network.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
            loss_gen=loss_gen.item(),
            rank=device,
            gpu_mem=(torch.cuda.memory_reserved(0)/1024/1024/1024)
        )
        loop.set_description(f"Epoch {epoch}")


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    loader = get_loader()

    ddp_gen = DDP(Generator(LOG_RESOLUTION, W_DIM).to(rank), device_ids=[rank])
    ddp_critic = DDP(Discriminator(LOG_RESOLUTION).to(rank), device_ids=[rank])
    ddp_mapping_network = DDP(MappingNetwork(Z_DIM, W_DIM).to(rank), device_ids=[rank])
    ddp_path_length_penalty = PathLengthPenalty(0.99).to(rank)

    opt_gen = optim.Adam(ddp_gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(ddp_critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_mapping_network = optim.Adam(ddp_mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    for epoch in range(EPOCHS):
        trainer(
            ddp_critic, ddp_gen, ddp_path_length_penalty, ddp_mapping_network,
            loader, opt_critic, opt_gen, opt_mapping_network, epoch, rank
        )
        if epoch % 10 == 0:
            generate_examples(ddp_gen, epoch, ddp_mapping_network, rank)
            save_everything(ddp_critic, ddp_gen, ddp_path_length_penalty, ddp_mapping_network,
                            opt_critic, opt_gen, opt_mapping_network, epoch)

    # create model and move it to GPU with id rank
    # model = ToyModel().to(rank)
    # ddp_model = DDP(model, device_ids=[rank])
    #
    # loss_fn = nn.MSELoss()
    # optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    #
    # optimizer.zero_grad()
    # outputs = ddp_model(torch.randn(20, 10))
    # labels = torch.randn(20, 5).to(rank)
    # loss_fn(outputs, labels).backward()
    # optimizer.step()

    cleanup()


def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
