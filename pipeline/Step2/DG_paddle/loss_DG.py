import paddle
import paddle.nn as nn
from loss_fn import HingeEmbeddingLoss
from model import Discriminator, Generator

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
generator_B.set_state_dict(discoGAN_ckpt['generator_B'])
discriminator_A.set_state_dict(discoGAN_ckpt['discriminator_B'])
discriminator_B.set_state_dict(discoGAN_ckpt['discriminator_B'])

generator_A.eval()
generator_B.eval()
discriminator_A.eval()
discriminator_B.eval()

A, _ = gen_fake_data(seed=42, shape=[4, 3, 64, 64])
B, _ = gen_fake_data(seed=100, shape=[4, 3, 64, 64])

with paddle.no_grad():

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

reprod_logger.add("gen_loss", gen_loss.cpu().detach().numpy())
reprod_logger.add("dis_loss", dis_loss.cpu().detach().numpy())
reprod_logger.save("loss_paddle.npy")
