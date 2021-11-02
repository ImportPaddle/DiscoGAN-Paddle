import numpy as np
import paddle
from paddle import nn


class HingeEmbeddingLoss(nn.Layer):
    """
         / x_i,                   if y_i == 1
    l_i =
         \ max(0, margin - x_i),  if y_i == -1
    """
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super(HingeEmbeddingLoss, self).__init__()
        self.loss = None
        self.margin = margin
        self.reduction = reduction

    def forward(self, x, y):
        self.loss = paddle.where(y == 1., x, paddle.maximum(paddle.to_tensor(0.), self.margin - x))
        if self.reduction == 'mean':
            return self.loss.mean()
        if self.reduction == 'sum':
            return self.loss.sum()
        else:
            raise ValueError(f"choose reduction from ['mean', 'sum'], but got {self.reduction}")

    # def forward(self, x, y):
    #     if (y == 1.).all():
    #         self.loss = x
    #     if (y == -1.).all():
    #         self.loss = paddle.maximum(paddle.to_tensor(0.), self.margin - x)
    #     if self.reduction == 'mean':
    #         return self.loss.mean()
    #     if self.reduction == 'max':
    #         return self.loss.max()
    #     else:
    #         raise ValueError(f"choose reduction from ['mean', 'max'], but got {self.reduction}")
