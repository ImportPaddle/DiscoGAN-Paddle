import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np

kernel_sizes = [4, 3, 3]
strides = [2, 2, 1]
paddings = [0, 0, 1]

latent_dim = 300


class Discriminator(nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, 4, 2, 1, bias_attr=False)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2D(64, 64 * 2, 4, 2, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(64 * 2)
        self.relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2D(64 * 2, 64 * 4, 4, 2, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(64 * 4)
        self.relu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2D(64 * 4, 64 * 8, 4, 2, 1, bias_attr=False)
        self.bn4 = nn.BatchNorm2D(64 * 8)
        self.relu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2D(64 * 8, 1, 4, 1, 0, bias_attr=False)

    def forward(self, input):
        conv1 = self.conv1(input)
        relu1 = self.relu1(conv1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)

        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3(bn3)

        conv4 = self.conv4(relu3)
        bn4 = self.bn4(conv4)
        relu4 = self.relu4(bn4)

        conv5 = self.conv5(relu4)

        return F.sigmoid(conv5), [relu2, relu3, relu4]


class Generator(nn.Layer):
    def __init__(self, extra_layers=False):

        super(Generator, self).__init__()

        if extra_layers == True:
            self.main = nn.Sequential(
                nn.Conv2D(3, 64, 4, 2, 1, bias_attr=False),
                nn.LeakyReLU(0.2),
                nn.Conv2D(64, 64 * 2, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 2),
                nn.LeakyReLU(0.2),
                nn.Conv2D(64 * 2, 64 * 4, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 4),
                nn.LeakyReLU(0.2),
                nn.Conv2D(64 * 4, 64 * 8, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 8),
                nn.LeakyReLU(0.2),
                nn.Conv2D(64 * 8, 100, 4, 1, 0, bias_attr=False),
                nn.BatchNorm2D(100),
                nn.LeakyReLU(0.2),

                nn.Conv2DTranspose(100, 64 * 8, 4, 1, 0, bias_attr=False),
                nn.BatchNorm2D(64 * 8),
                nn.ReLU(),
                nn.Conv2DTranspose(64 * 8, 64 * 4, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 4),
                nn.ReLU(),
                nn.Conv2DTranspose(64 * 4, 64 * 2, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 2),
                nn.ReLU(),
                nn.Conv2DTranspose(64 * 2, 64, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64),
                nn.ReLU(),
                nn.Conv2DTranspose(64, 3, 4, 2, 1, bias_attr=False),
                nn.Sigmoid()
            )

        if extra_layers == False:
            self.main = nn.Sequential(
                nn.Conv2D(3, 64, 4, 2, 1, bias_attr=False),
                nn.LeakyReLU(0.2),
                nn.Conv2D(64, 64 * 2, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 2),
                nn.LeakyReLU(0.2),
                nn.Conv2D(64 * 2, 64 * 4, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 4),
                nn.LeakyReLU(0.2),
                nn.Conv2D(64 * 4, 64 * 8, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 8),
                nn.LeakyReLU(0.2),

                nn.Conv2DTranspose(64 * 8, 64 * 4, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 4),
                nn.ReLU(True),
                nn.Conv2DTranspose(64 * 4, 64 * 2, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64 * 2),
                nn.ReLU(True),
                nn.Conv2DTranspose(64 * 2, 64, 4, 2, 1, bias_attr=False),
                nn.BatchNorm2D(64),
                nn.ReLU(True),
                nn.Conv2DTranspose(64, 3, 4, 2, 1, bias_attr=False),
                nn.Sigmoid()
            )

    def forward(self, input):
        return self.main(input)
