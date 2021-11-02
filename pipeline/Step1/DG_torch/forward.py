import numpy as np
import torch
import sys
sys.path.append(r"D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\pipeline")
from model import Discriminator, Generator
from utils import gen_fake_data, paddle2torch
from reprod_log import ReprodLogger

if __name__ == "__main__":

    # def logger
    reprod_logger = ReprodLogger()

    generator = Generator()
    discriminator = Discriminator()

    generator.load_state_dict(torch.load("./torch_gen.pth"))
    discriminator.load_state_dict(torch.load("./torch_dis.pth"))
    generator.eval()
    discriminator.eval()

    _, fake_data = gen_fake_data(shape=[4, 3, 64, 64])

    # forward
    with torch.no_grad():
        dis_out = discriminator(fake_data)
        gen_out = generator(fake_data)

    reprod_logger.add("dis_out", dis_out.cpu().detach().numpy())
    reprod_logger.add("gen_out", gen_out.cpu().detach().numpy())

    reprod_logger.save("forward_torch.npy")
