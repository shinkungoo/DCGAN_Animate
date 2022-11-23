import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channel_number, discriminator_filter_number):
        super(Discriminator, self).__init__()
        # 32 x 32 子网
        self.layer1 = nn.Sequential(nn.Conv2d(channel_number, discriminator_filter_number, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(discriminator_filter_number),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16 子网
        self.layer2 = nn.Sequential(nn.Conv2d(discriminator_filter_number, discriminator_filter_number * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(discriminator_filter_number * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8 子网
        self.layer3 = nn.Sequential(nn.Conv2d(discriminator_filter_number * 2, discriminator_filter_number * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(discriminator_filter_number * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4 子网
        self.layer4 = nn.Sequential(nn.Conv2d(discriminator_filter_number * 4, 1, kernel_size=4, stride=1, padding=0))

    # 构建神经网络
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out