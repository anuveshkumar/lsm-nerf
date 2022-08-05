import torch
import torch.nn as nn

TINY_NUMBER = 1e-6


def img2mse(x, y, mask=None):
    """
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional [(...)]
    :return: mse score
    """
    if mask is None:
        return torch.mean((x - y) * (x - y))

    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, outputs, ray_batch):
        pred_rgb = outputs['rgb']
        pred_mask = None
        gt_rgb = ray_batch['rgb']

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss