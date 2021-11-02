import numpy as np
import paddle
import sys
sys.path.append(r"D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\pipeline")
from model import Discriminator, Generator
from utils import gen_fake_data, torch2paddle
from reprod_log import ReprodLogger

if __name__ == "__main__":
    paddle.set_device("cpu")
    # load model
    # gen_ckpt = paddle.load(r"D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\datasets\ckpts\model_gen_A_3.0.pdparams")
    # dis_ckpt = paddle.load(r"D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\datasets\ckpts\model_dis_A-3.0.pdparams")

    # def logger
    reprod_logger = ReprodLogger()

    generator = Generator()
    discriminator = Discriminator()

    gen_ckpt = torch2paddle(r"D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\pipeline\Step1\DG_torch\torch_gen.pth")
    dis_ckpt = torch2paddle(r"D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\pipeline\Step1\DG_torch\torch_dis.pth")

    generator.set_state_dict(gen_ckpt)
    discriminator.set_state_dict(dis_ckpt)
    generator.eval()
    discriminator.eval()

    fake_data, _ = gen_fake_data(shape=[4, 3, 64, 64])

    # forward
    with paddle.no_grad():
        dis_out = discriminator(fake_data)
        gen_out = generator(fake_data)
    #
    reprod_logger.add("dis_out", dis_out.cpu().detach().numpy())
    reprod_logger.add("gen_out", gen_out.cpu().detach().numpy())

    reprod_logger.save("forward_paddle.npy")
