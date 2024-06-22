# from trainer2 import train_wgan
from WGAN.tester import W_GAN_MNIST

if __name__ == '__main__':
    # model_path = "./models/"
    # print(torch.cuda.is_available())
    # print('\ndevice name:\n', torch.cuda.get_device_name())
    # print('\ncapability:\n', torch.cuda.get_device_capability())

    # train_WGAN(dataloader)

    W_GAN_MNIST()
    
    # batch_size = 64
    # dataloader = load_data(download=False, batch_size=batch_size, MNIST=False,
    #                        show_samples=False, num_samples=10, size=(256, 256))
    #
    # D = Discriminator256()
    # G = Generator256()
    # y_G, y_D = train_WGAN(G, D, dataloader, latent_dim=64, batch_size=batch_size, device='cuda', g_lr=0.01, d_lr=0.01, n_epochs=1)
    #
    # save_graph(y_G, y_D)
    #
    # G.to('cpu')
    # D.to('cpu')
    # G, D = G.eval(), D.eval()
    #
    # fake_samples = generate_samples(G, 'final')
    #
    # # save  models
    # torch.jit.save(torch.jit.trace(G, torch.rand(batch_size, 64, 1, 1)), 'results/G_model256.pth')
    # torch.jit.save(torch.jit.trace(D, fake_samples), 'results/D_model256.pth')

