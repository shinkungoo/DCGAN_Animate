# 这个文件是使用已经训练好的模型生成图片的，使用LSGAN_waife.ipynb 可以生成模型

import Generator
import Discriminator
import Initialization
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.utils as vu

from torch.autograd import Variable
from torchvision import transforms

Initialization.cuda_test()
opt = Initialization.parser_init()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

data_transform = transforms.Compose([
    transforms.Resize(opt.imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
