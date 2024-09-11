import torch
from ignite.metrics import FID, InceptionScore
from PIL import Image, ImageFile
import torchvision.transforms.v2 as transforms
from StyleGAN.classes import Generator as GenS1
from StyleGAN2.classes import Generator as GenS2
from StyleGAN2.classes import MappingNetwork
from torch.utils.data import DataLoader
from torchvision import datasets
from ignite.engine import Engine
import numpy

def get_loader(image_size=256):
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            )
        ]
    )
    # batch_size = batch_size
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    # take subset of data such that batches are always whole.
    dataset_subset = torch.utils.data.Subset(dataset, numpy.random.choice(len(dataset),
                                             ((len(dataset)//batch_size) * batch_size), replace=False))

    loader = DataLoader(
        dataset_subset,
        num_workers=6,
        batch_size=batch_size,
        shuffle=True,
    )
    return loader, dataset


def interpolate(batch):
    arr = []
    # print(batch.size())
    for img in batch:
        # print(len(img))
        # if len(img) == 2:
        #     print(img)
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299, 299), Image.Resampling.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


def evaluation_step(engine, batch):
    with torch.no_grad():
        # noise = torch.randn(batch_size, 256)
        noise = get_noise(batch_size)
        netG.eval()
        # fake_batch = netG(noise, 1, 6)
        w = get_w(batch_size)
        fake_batch = netG(w, noise)
        fake = interpolate(fake_batch)
        # print(len(batch))
        # print(batch)
        # bt = next(iter(batch))
        # print('\nthis one', bt[0].size(), batch)
        real = interpolate(batch[0])
        print('step')
        return fake, real

def get_noise(batch_size):
    noise = []
    resolution = 4
    for i in range(LOG_RESOLUTION):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        noise.append((n1, n2))
        resolution *= 2
    return noise

def log_training_results():
    evaluator.run(test_dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']
    fid_values.append(fid_score)
    is_values.append(is_score)
    print(f"*   FID : {fid_score:4f}")
    print(f"*    IS : {is_score:4f}")


def get_w(batch_size=4):
    z = torch.randn(batch_size, W_DIM)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    batch_size = 4
    device = 'cpu'
    # DATASET = "/vol/bitbucket/ik323/fyp/graffiti"
    DATASET = "./data/datasetEvaluation/"
    LEARNING_RATE = 1e-3
    CHANNELS_IMG = 3
    Z_DIM = 256
    # latent_dim = z_dim
    latent_dim = 256
    W_DIM = 256
    IN_CHANNELS = 256
    LOG_RESOLUTION = 8

    test_dataloader, _ = get_loader()

    # ./results/StyleGAN1/archive-v1.0.0/models/step6_alpha1/trained.pth
    # netG = GenS1(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(device)
    # model = torch.load("./results/StyleGAN1/archive-v1.0.0/models/step6_alpha1/trained.pth")
    # netG.load_state_dict(model["generator"])

    # ./results/usageModels/styleGAN2/epoch100/trained.pth
    netG = GenS2(LOG_RESOLUTION, W_DIM).to(device)
    model = torch.load("./results/usageModels/styleGAN2/epoch100/trained.pth")
    netG.load_state_dict(model["generator"])
    mapping_network = MappingNetwork(Z_DIM, W_DIM)
    mapping_network.load_state_dict(model["mapping"])

    fid_values = []
    is_values = []

    fid_metric = FID()
    is_metric = InceptionScore(output_transform=lambda x: x[0])

    evaluator = Engine(evaluation_step)
    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator, "is")

    log_training_results()


