import torch
import argparse


def cuda_test():
    print('CUDA版本:', torch.version.cuda)
    print('Pytorch版本:', torch.__version__)
    print('显卡是否可用:', '可用' if (torch.cuda.is_available()) else '不可用')
    print('显卡数量:', torch.cuda.device_count())
    print('是否支持BF16数字格式:', '支持' if (torch.cuda.is_bf16_supported()) else '不支持')
    print('当前显卡型号:', torch.cuda.get_device_name())
    print('当前显卡的CUDA算力:', torch.cuda.get_device_capability())


def parser_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help="generator filter size")
    parser.add_argument('--ndf', type=int, default=64, help="discriminator filter size")
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
    parser.add_argument('--outf', default='output/', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    opt = parser.parse_args(args=[])
    return opt
