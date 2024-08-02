import os
from PIL import Image

def is_valid_image_pillow(file_name):
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False

data_path = "/vol/bitbucket/ik323/fyp/images/"

count = 0
print('Verifying...')
for filename in os.scandir(data_path):
    if not is_valid_image_pillow(filename.path):
        print('Invalid. Deleting: ', filename.path)
        os.remove(filename.path)
    count += 1
    if count % 1000 == 0:
        print(f'Images checked: {count}, Checkpoint file: {filename.path}')

print('Verified')
