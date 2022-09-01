import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_classes, image_size=256, slope=0.2):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # input: (b, 3, 256, 256)
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), # NOTE: maybe layer norm, see: https://github.com/mlberkeley/Creative-Adversarial-Networks/blob/master/discriminators.py
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 32, 128, 128)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 64, 64, 64)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 128, 32, 32)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 256, 16, 16)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 512, 8, 8) 
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # (b, 512, 4, 4)
        )

        self.disc = nn.Sequential(
            nn.Linear(512*4*4, 1),
            nn.Sigmoid(),
        )

        self.cls = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(512, self.num_classes),
        )
        
    def forward(self, x):
        assert x.shape[1:] == (3, 256, 256)
        out = self.main(x).flatten(start_dim=1)
        real_prob = self.disc(out)
        cls_logits = self.cls(out)
        return real_prob, cls_logits


if __name__ == '__main__':
    from torchinfo import summary 
    model = Discriminator(num_classes=10)
    summary(model, (2, 3, 256, 256))
