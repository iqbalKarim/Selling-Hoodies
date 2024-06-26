from WGAN.tester import W_GAN_MNIST, W_GAN_SCULPTURES
from WGAN.MNISTClasses import Critic, Generator, Critic_Comp, Generator_Comp, Generator_Insp, Critic_Inps
from pytorchsummary import summary


def paramsearch_and_training_MNIST(complicated=False):
    # for batch in [128]:
    for batch in [64, 128]:
        for g_lr, d_lr in [(0.0001, 0.0001), (0.0002, 0.0001), (0.0001, 0.0002), (0.0002, 0.0002)]:
            print(f"Training for: \n "
                  f"\tBatch Size: {batch} \n"
                  f"\tGen LR: {g_lr}\n"
                  f"\tDisc LR: {d_lr}")
            W_GAN_MNIST(batch, g_lr, d_lr, complicated=complicated)

def paramsearch_and_training_sculptures():
    for batch in [256]:
        for g_lr, d_lr in [(0.0001, 0.0001), (0.0002, 0.0001), (0.0001, 0.0002), (0.0002, 0.0002)]:
            print(f"Training for: \n "
                  f"\tBatch Size: {batch} \n"
                  f"\tGen LR: {g_lr}\n"
                  f"\tDisc LR: {d_lr}")
            W_GAN_SCULPTURES(batch, g_lr, d_lr)

if __name__ == '__main__':
    # model_path = "./models/"
    # print(torch.cuda.is_available())
    # print('\ndevice name:\n', torch.cuda.get_device_name())
    # print('\ncapability:\n', torch.cuda.get_device_capability())

    # paramsearch_and_training_MNIST(complicated=True)
    # paramsearch_and_training_sculptures()
    # D, G = Critic_Inps(), Generator_Insp()
    #

    W_GAN_MNIST(D=Critic_Inps(), G=Generator_Insp(), z_dim=100)
