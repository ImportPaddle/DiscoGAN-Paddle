import numpy as np

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from loss_fn import HingeEmbeddingLoss
from model import Discriminator, Generator

from itertools import chain
import sys
sys.path.append(r"D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\pipeline")
from utils import gen_fake_data, torch2paddle, torch2paddle_gether
from reprod_log import ReprodLogger

recon_criterion = nn.MSELoss()
gan_criterion = nn.BCELoss()
feat_criterion = HingeEmbeddingLoss()

# def logger
reprod_logger = ReprodLogger()


def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion(l2, paddle.ones(l2.shape))
        losses += loss

    return losses


def get_gan_loss(dis_real, dis_fake, criterion):
    labels_dis_real = paddle.ones([dis_real.shape[0], 1])
    labels_dis_fake = paddle.zeros([dis_fake.shape[0], 1])
    labels_gen = paddle.ones([dis_fake.shape[0], 1])

    dis_real = dis_real.reshape([dis_real.shape[0], 1])
    dis_fake = dis_fake.reshape([dis_fake.shape[0], 1])
    dis_loss = criterion(dis_real, labels_dis_real) * 0.5 + criterion(dis_fake, labels_dis_fake) * 0.5
    gen_loss = criterion(dis_fake, labels_gen)

    return dis_loss, gen_loss


generator_A = Generator()
generator_B = Generator()
discriminator_A = Discriminator()
discriminator_B = Discriminator()

# 加载预训练模型
discoGAN_ckpt = torch2paddle_gether("./discoGAN.pth")
generator_A.set_state_dict(discoGAN_ckpt['generator_A'])
generator_B.set_state_dict(discoGAN_ckpt['generator_A'])
discriminator_A.set_state_dict(discoGAN_ckpt['discriminator_B'])
discriminator_B.set_state_dict(discoGAN_ckpt['discriminator_B'])

generator_A.eval()
generator_B.eval()
discriminator_A.eval()
discriminator_B.eval()

gen_params = chain(generator_A.parameters(), generator_B.parameters())
dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

# 优化器
optim_gen = optim.Adam(parameters=gen_params,
                       learning_rate=0.0002,
                       beta1=0.5, beta2=0.999,
                       weight_decay=0.00001)
optim_dis = optim.Adam(parameters=dis_params,
                       learning_rate=0.0002,
                       beta1=0.5, beta2=0.999,
                       weight_decay=0.00001)

A, _ = gen_fake_data(seed=42, shape=[4, 3, 64, 64])
B, _ = gen_fake_data(seed=100, shape=[4, 3, 64, 64])

def main(iters):
    gen_loss_lst = []
    dis_loss_lst = []
    for iter in range(iters):
        AB = generator_B(B)
        BA = generator_A(A)

        ABA = generator_A(AB)
        BAB = generator_B(BA)

        # Reconstruction Loss
        recon_loss_A = recon_criterion(ABA, A)
        recon_loss_B = recon_criterion(BAB, B)

        # Real/Fake GAN Loss (A)
        A_dis_real, A_feats_real = discriminator_A(A)
        A_dis_fake, A_feats_fake = discriminator_A(BA)

        dis_loss_A, gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, gan_criterion)
        fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_criterion)

        # Real/Fake GAN Loss (B)
        B_dis_real, B_feats_real = discriminator_B(B)
        B_dis_fake, B_feats_fake = discriminator_B(AB)

        dis_loss_B, gen_loss_B = get_gan_loss(B_dis_real, B_dis_fake, gan_criterion)
        fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, feat_criterion)

        rate = 0.01

        gen_loss_A_total = (gen_loss_B * 0.1 + fm_loss_B * 0.9) * (1. - rate) + recon_loss_A * rate
        gen_loss_B_total = (gen_loss_A * 0.1 + fm_loss_A * 0.9) * (1. - rate) + recon_loss_B * rate

        gen_loss = gen_loss_A_total + gen_loss_B_total
        dis_loss = dis_loss_A + dis_loss_B
        print(f"paddle gen loss: {gen_loss.item()}, dis loss: {dis_loss.item()}")

        # 反传
        if iter % 2 == 0:
            dis_loss.backward()
            optim_dis.step()
            optim_dis.clear_grad()
        else:
            gen_loss.backward()
            optim_gen.step()
            optim_gen.clear_grad()

        gen_loss_lst.append(gen_loss)
        dis_loss_lst.append(dis_loss)

    for idx, loss in enumerate(gen_loss_lst):
        print(f"torch gen loss {idx}: {loss.item()}")
        reprod_logger.add(f"gen_loss_{idx}", loss.cpu().detach().numpy())
    for idx, loss in enumerate(dis_loss_lst):
        print(f"torch dis loss {idx}: {loss.item()}")
        reprod_logger.add(f"dis_loss_{idx}", loss.cpu().detach().numpy())
    reprod_logger.save("bp_paddle.npy")


if __name__ == '__main__':
    main(iters=5)
