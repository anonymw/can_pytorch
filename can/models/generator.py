import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz=100, slope=0.2, init_nc=1024):
        super(Generator, self).__init__()
        self.init_nc = init_nc

        # NOTE: maybe 2048x4x4, see page 9 of paper
        self.linear = nn.Linear(nz, self.init_nc * 4 * 4)
        # then reshape to (b, 1024, 4, 4)
        self.main = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 1024, 4, 4)
            nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 1024, 8, 8)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 512, 16, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 256, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 128, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 64, 128, 128)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
            # (b, 3, 256, 256)
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(-1, self.init_nc, 4, 4)
        return self.main(out)


if __name__ == '__main__':
    from torchinfo import summary
    model = Generator()
    summary(model, (2, 100))