import os
from PIL import Image

data_path = "./datasetEvaluation/"

if __name__ == "__main__":
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            input_path = os.path.join(root, name)
            image = Image.open(input_path)
            print(input_path, ': ', image.mode)
            if image.mode != 'RGB':
                print(input_path, ': ', image.mode)
                newImg = image.convert('RGB')
                newImg.save(fp=input_path)
# ./dataset/train2\1.jpg  :  L
# ./dataset/train2\5.jpg  :  L