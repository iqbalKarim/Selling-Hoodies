import urllib.request
import os
import pandas as pd
from PIL import Image

def is_valid_image_pillow(file_name):
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False

with open('out.txt', 'w') as f:
    print("Loading data.\n", file=f)

    # data_path = "./data/dataset/train/"
    data_path = "/vol/bitbucket/ik323/fyp/dataset/train/"

    count = 0
    df = pd.read_csv("omniart_v3_datadump.csv")
    # only get sculptures
    df = df[df["artwork_type"] == "sculpture"]

    image_urls = df["image_url"]
    # print(image_urls)
    print("Data dump initialized.\n", file=f)

    if not os.path.exists(data_path):
        print("Directory does not exist. \nCreating directory ....", end=" ", file=f)
        os.makedirs(data_path)
        print("Created!\n", file=f)
    else:
        print("Directory already exists.\n", file=f)

    ######
    # This is to prevent 403 Forbidden error caused by missing user-agent
    # link: https://stackoverflow.com/questions/69782728/urllib-error-httperror-http-error-403-forbidden-with-urllib-requests#:~:text=This%20means%20some%20URLs%20just,the%20URL%20in%20your%20script.
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'MyApp/1.0')]
    urllib.request.install_opener(opener)
    #####

    print('Downloading images.', file=f)
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
            print('Error on ' + str(count) + ' with URL: ' + url + '\n\t' + repr(e), file=f)
        count += 1
    print("Images downloaded and verified.\n", file=f)
