import os
from PIL import Image

def is_valid_image_pillow(file_name):
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False

data_path = "/vol/bitbucket/ik323/fyp/dataset/train/"

for filename in os.scandir(data_path):
    if not is_valid_image_pillow(filename.path):
        print('Invalid. Deleting: ', filename.path)
        os.remove(filename.path)

print('Verified')
