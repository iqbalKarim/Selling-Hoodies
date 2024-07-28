import os
import pandas as pd
import matplotlib.pyplot as plt


CHARACTER_COUNT = 26
data = pd.read_csv('/vol/bitbucket/ik323/fyp/A_Z Handwritten Data.csv')
mapping = {str(i): chr(i + 65) for i in range(26)}

def generate_dataset(folder, end, start=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"The folder '{folder}' has been created successfully!")
    else:
        print(f"The folder '{folder}' already exists.")

    for i in range(CHARACTER_COUNT):
        dd = data[data['0'] == i]
        for j in range(start, end):
            ddd = dd.iloc[j]
            x = ddd[1:].values
            x = x.reshape((28, 28))
            plt.axis('off')
            plt.imsave(f'{folder}/{mapping[str(i)]}_{j}.jpg', x, cmap='binary_r')


generate_dataset('/vol/bitbucket/ik323/fyp/letters/train', 1100)

