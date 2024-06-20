from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.g = nn.Sequential(
            # input: 64 x 1 x 1

            nn.ConvTranspose2d(64, 128, kernel_size=6, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 64 x 1 x 1 ---> 128 x 4 x 4

            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 128 x 4 x 4 ---> 256 x 8 x 8

            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 256 x 8 x 8 ---> 512 x 16 x 16

            nn.ConvTranspose2d(512, 256, kernel_size=6, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 512 x 16 x 16 ---> 256 x 64 x 64

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 256 x 64 x 64 ---> 128 x 128 x 128

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 128 x 128 x 128 ---> 64 x 256 x 256

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 64 x 256 x 256 ---> 3 x 512 x 512
        )

    def forward(self, z):
        out = self.g(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d = nn.Sequential(
            # input: 3 x 512 x 512
            nn.Conv2d(3, 64, kernel_size=6, stride=4, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 3 x 512 x 512 ---> 64 x 128 x 128

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 128 x 128 ---> 128 x 64 x 64

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 64 x 64 ---> 256 x 16 x 16

            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 16 x 16 ---> 128 x 8 x 8

            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 8 x 8 ---> 64 x 4 x 4

            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # 64 x 4 x 4 ---> 1 x 1 x 1
        )

    def forward(self, x):
        out = self.d(x)
        return out


class Generator256(nn.Module):
    def __init__(self):
        super(Generator256, self).__init__()

        self.g = nn.Sequential(
            # input: 64 x 1 x 1

            nn.ConvTranspose2d(64, 128, kernel_size=6, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 64 x 1 x 1 ---> 128 x 4 x 4

            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 128 x 4 x 4 ---> 256 x 8 x 8

            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 256 x 8 x 8 ---> 512 x 16 x 16

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 512 x 16 x 16 ---> 256 x 32 x 32

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 256 x 32 x 32 ---> 128 x 64 x 64

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 128 x 64 x 64 ---> 64 x 128 x 128

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 64 x 128 x 128 ---> 3 x 256 x 256
        )

    def forward(self, z):
        out = self.g(z)
        return out


class Discriminator256(nn.Module):
    def __init__(self):
        super(Discriminator256, self).__init__()

        self.d = nn.Sequential(
            # input: 3 x 256 x 256
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 3 x 256 x 256 ---> 64 x 128 x 128

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 128 x 128 ---> 128 x 64 x 64

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 64 x 64 ---> 256 x 16 x 16

            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 16 x 16 ---> 128 x 8 x 8

            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 8 x 8 ---> 64 x 4 x 4

            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # 64 x 4 x 4 ---> 1 x 1 x 1
        )

    def forward(self, x):
        out = self.d(x)
        return out
