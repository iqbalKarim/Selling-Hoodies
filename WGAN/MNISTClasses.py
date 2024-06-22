from torch import nn


class Critic(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=16):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            self.get_critic_block(im_chan,
                                  hidden_dim * 4,
                                  kernel_size=4,
                                  stride=2),

            self.get_critic_block(hidden_dim * 4,
                                  hidden_dim * 8,
                                  kernel_size=4,
                                  stride=2, ),

            self.get_critic_final_block(hidden_dim * 8,
                                        1,
                                        kernel_size=4,
                                        stride=2, ),

        )

    def get_critic_block(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def get_critic_final_block(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
        )

    def forward(self, image):
        return self.disc(image)


class Generator(nn.Module):

    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()

        self.z_dim = z_dim

        self.gen = nn.Sequential(

            self.get_generator_block(z_dim,
                                     hidden_dim * 4,
                                     kernel_size=3,
                                     stride=2),

            self.get_generator_block(hidden_dim * 4,
                                     hidden_dim * 2,
                                     kernel_size=4,
                                     stride=1),

            self.get_generator_block(hidden_dim * 2,
                                     hidden_dim,
                                     kernel_size=3,
                                     stride=2,
                                     ),

            self.get_generator_final_block(hidden_dim,
                                           im_chan,
                                           kernel_size=4,
                                           stride=2)

        )

    def get_generator_block(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

    def get_generator_final_block(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.Tanh()
        )

    def forward(self, noise):
        # x = noise.view(noise.shape[0], self.z_dim, 1, 1)
        return self.gen(noise)