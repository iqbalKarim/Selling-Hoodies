from PIL import Image
from NST.trainer import run_style_transfer
import torchvision.transforms as transforms
import torch

def image_loader(img):
    loader = transforms.Compose([
        transforms.Resize((512, 512)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = loader(img).unsqueeze(0)
    return image.to(device, torch.float)

def convert_image(style_img, content_img, norm_mean, norm_std, cnn, num_steps=300):
    unloader = transforms.ToPILImage()

    style_img.show()
    content_img.show()

    style_img = image_loader(style_img)
    content_img = image_loader(content_img)
    input_img = content_img.clone()
    print('running transfer')
    output = run_style_transfer(cnn, norm_mean, norm_std,
                                content_img, style_img, input_img,
                                num_steps=num_steps)
    image = output.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.show()
    return image

if __name__ == '__main__':
    print('run')