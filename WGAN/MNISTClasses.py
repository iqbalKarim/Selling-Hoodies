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


class Generator_Insp(nn.Module):
    def __init__(self):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.g = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=2, stride=2, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.g(x)


class Critic_Insp(nn.Module):
    def __init__(self):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.c = nn.Sequential(
            # Image (Cx28x28)
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=2, stride=2, padding=2),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.c(x)