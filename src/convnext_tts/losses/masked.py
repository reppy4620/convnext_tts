import torch.nn.functional as F


def masked_mse_loss(x, y, mask):
    loss = (x - y).pow(2) * mask
    loss = loss.sum() / mask.sum()
    return loss


def masked_l1_loss(x, y, mask):
    loss = (x - y).abs() * mask
    loss = loss.sum() / mask.sum()
    return loss


def masked_bce_loss(x, y, mask):
    loss = F.binary_cross_entropy_with_logits(x, y, reduction="none") * mask
    loss = loss.sum() / mask.sum()
    return loss
