import urllib.request
import os
import pandas as pd
import torchvision.datasets
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageFile


# ['id,artwork_name,artist_full_name,artist_first_name,artist_last_name,creation_year,century,source_url,image_url,collection_origins,artwork_type,school,original_id_in_collection,created_at,last_modified,omni_id,created_by_id,general_type,geocoded,color_pallete,dominant_color,palette_count\n']

def is_valid_image_pillow(file_name):
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False


def plot_transformed_images(transform):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        transform (PyTorch Transforms): Transforms to apply to images.
    """
    print('Previewing data.')
    random_image_paths = [
        "./data/dataset/5.jpg", "./data/dataset/41.jpg", "./data/dataset/16.jpg"
    ]
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path}", fontsize=16)
    plt.show()


def load_data(download=False, batch_size=64, MNIST=False, num_samples=5, show_samples=False, size=(512,512)):
    print("Loading data.\n")
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if MNIST:
        data = torchvision.datasets.MNIST("/vol/bitbucket/ik323/fyp/mnist/", train=True, 
                                          transform=transforms.ToTensor(), download=True)
        # print(len(data))
        return DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6)

    # data_path = "./data/dataset/train/"
    data_path = "/vol/bitbucket/ik323/fyp/dataset/train/"

    if download:

        count = 0
        df = pd.read_csv("omniart_v3_datadump.csv")
        # only get sculptures
        df = df[df["artwork_type"] == "sculpture"]

        image_urls = df["image_url"]
        # print(image_urls)
        print("Data dump initialized.\n")

        if not os.path.exists(data_path):
            print("Directory does not exist. \nCreating directory ....", end=" ")
            os.makedirs(data_path)
            print("Created!\n")
        else:
            print("Directory already exists.\n")

        ######
        # This is to prevent 403 Forbidden error caused by missing user-agent
        # link: https://stackoverflow.com/questions/69782728/urllib-error-httperror-http-error-403-forbidden-with-urllib-requests#:~:text=This%20means%20some%20URLs%20just,the%20URL%20in%20your%20script.
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'MyApp/1.0')]
        urllib.request.install_opener(opener)
        #####

        print('Downloading images.')
        # while count < 50000:
        for url in image_urls:
            # print(url)
            try:
                path_to_file = data_path + str(count) + ".jpg"
                # urllib.request.urlretrieve(image_urls[count], path_to_file)
                urllib.request.urlretrieve(url, path_to_file)
                if not is_valid_image_pillow(path_to_file):
                    os.remove(path_to_file)
            except Exception as e:
                # print('Error on ' + str(count) + ' with URL: ' + image_urls[count] + '\n\t' + repr(e))
                print('Error on ' + str(count) + ' with URL: ' + url + '\n\t' + repr(e))
            count += 1
        print("Images downloaded and verified.\n")

    data_transform = transforms.Compose([
        # Resize the images to 512x512 (default) or size tuple
        transforms.Resize(size=size),
        # Turn the image into a torch.Tensor
        transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    ])
    # plot_transformed_images(data_transform)

    train_data = datasets.ImageFolder(root="/vol/bitbucket/ik323/fyp/dataset",
                                      transform=data_transform)
    # train_data = datasets.ImageFolder(root="./data/dataset/",
    #                                   transform=data_transform)

    print(f"\nTraining dataset: \n{train_data}")

    if show_samples:
        # images = train_data[0][0]
        for i in range(num_samples):
            img = train_data[i][0]                  #[color_channels, height, width]
            img_permute = img.permute(1, 2, 0)      #[height, width, color_channels]
            plt.figure(figsize=(10, 7))
            plt.imshow(img_permute)
            plt.axis("off")
            plt.title(f'Image: {i}', fontsize=14)
            plt.show()

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,  # how many samples per batch?
                                  num_workers=6,  # how many subprocesses to use for data loading? (higher = more)
                                  shuffle=True)
    print(f"Return training dataloader: \n\t{train_dataloader}")
    return train_dataloader
