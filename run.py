# 这个文件是使用已经训练好的模型生成图片的，使用LSGAN_waife.ipynb 可以生成模型
# 限制于我的比较差的电脑，大模型崩溃了太多次了，所以就只存储了小模型

import Generator
import Discriminator

import Initialization
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vu

from torch.autograd import Variable

if __name__ == "__main__":
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

    generator = Generator.Generator(3, opt.ngf, opt.nz)
    discriminator = Discriminator.Discriminator(3, opt.ndf)
    noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
    noise = Variable(noise)
    if opt.cuda and torch.cuda.is_available():
        print("cuda is available")
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        noise = noise.cuda()

    generator_state = torch.load("./trainedModel/netG.pth")
    discriminator_state = torch.load("./trainedModel/netD.pth")

    generator.load_state_dict(generator_state)
    discriminator.load_state_dict(discriminator_state)

    noise.data.normal_(0, 1)
    fake = generator(noise)
    vu.save_image(fake.data, 'output.png', normalize=True)
    print("generated successfully!")

