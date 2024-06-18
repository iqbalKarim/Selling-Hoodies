from load_data import load_data
from trainer import train_WGAN


if __name__ == '__main__':
    model_path = "./models/"

    # with open("./data/omniart_v3_datadump.csv") as input_file:
    #     head = [next(input_file) for _ in range(1)]
    # print(head)

    # df = pd.read_csv("./data/omniart_v3_datadump.csv")
    # image_urls = df["image_url"].tolist()
    # # image_urls.to_csv("./data/image_urls.csv", index=False)
    # with open("./data/image_urls.csv", 'w', newline='') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(image_urls)

    batch_size = 64
    dataloader = load_data(download=False, batch_size=batch_size)
    train_WGAN(dataloader)