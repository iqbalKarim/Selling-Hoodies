import os
from PIL import Image


if __name__ == "__main__":
    for root, dirs, files in os.walk("/vol/bitbucket/ik323/fyp/dataset/", topdown=False):
        for name in files:
            input_path = os.path.join(root, name)
            image = Image.open(input_path)
            if image.mode != 'RGB':
                print(input_path, ': ', image.mode)
                newImg = image.convert('RGB')
                newImg.save(fp=input_path)
# ./dataset/train2\1.jpg  :  L
# ./dataset/train2\5.jpg  :  L