import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, channel_number, generator_filter_number, latent_z):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(latent_z, generator_filter_number * 4, kernel_size=4),
                                    nn.BatchNorm2d(generator_filter_number * 4),
                                    nn.ReLU())
        # 4 x 4 子网
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(generator_filter_number * 4, generator_filter_number * 2, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(generator_filter_number * 2),
            nn.ReLU())
        # 8 x 8 子网
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(generator_filter_number * 2, generator_filter_number, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(generator_filter_number),
            nn.ReLU())

        # 16 x 16 子网
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(generator_filter_number, channel_number, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    # 构建神经网络
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out