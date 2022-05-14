import torch
from torch import nn


class FocalLossWithLogitsNegLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def extra_repr(self):
        return 'alpha={}, gamma={}'.format(self.alpha, self.gamma)

    def forward(self, pred, target):
        sigmoid_pred = pred.sigmoid()
        log_sigmoid = torch.nn.functional.logsigmoid(pred)
        loss = (target == 1) * self.alpha * torch.pow(1. - sigmoid_pred, self.gamma) * log_sigmoid

        log_sigmoid_inv = torch.nn.functional.logsigmoid(-pred)
        loss += (target == 0) * (1 - self.alpha) * torch.pow(sigmoid_pred, self.gamma) * log_sigmoid_inv

        return -loss


class DistillFocalLossWithLogitsNegLoss(nn.Module):
    def __init__(self, alpha, gamma, t=1.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.T = t

    def extra_repr(self):
        return 'alpha={}, gamma={}'.format(self.alpha, self.gamma)

    def forward(self, pred, target, guide):
        weight = torch.zeros_like(target)
        weight[target == 0] = 1. - self.alpha  # negative
        weight[target > 1e-5] = self.alpha # positive

        sigmoid_pred = pred.sigmoid()
        sigmoid_guide = (guide / self.T).sigmoid()

        log_sigmoid = torch.nn.functional.logsigmoid(pred)
        log_sigmoid_inv = torch.nn.functional.logsigmoid(-pred)

        coef = weight * torch.pow((sigmoid_pred - target).abs(), self.gamma)
        loss = sigmoid_guide * log_sigmoid + (1. - sigmoid_guide) * log_sigmoid_inv
        loss = (coef * loss).sum()

        return -loss


class FocalLossWithLogitsNegSoftLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def extra_repr(self):
        return 'alpha={}, gamma={}'.format(self.alpha, self.gamma)

    def forward(self, pred, target):

        weight = torch.zeros_like(target)
        weight[target == 0] = 1. - self.alpha  # negative
        weight[target > 1e-5] = self.alpha # positive

        sigmoid_pred = pred.sigmoid()

        log_sigmoid = torch.nn.functional.logsigmoid(pred)
        log_sigmoid_inv = torch.nn.functional.logsigmoid(-pred)

        coef = weight * torch.pow((sigmoid_pred - target).abs(), self.gamma)
        loss = target * log_sigmoid + (1. - target) * log_sigmoid_inv
        loss = (coef * loss).sum()

        return -loss


class FocalSmoothBCEWithLogitsNegLoss(nn.Module):
    def __init__(self, alpha, gamma, pos, neg):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos = pos
        self.neg = neg

    def forward(self, logits, target):
        target_prob = target.clone().float()
        target_prob[target == 1] = self.pos
        target_prob[target == 0] = self.neg

        sigmoid_pred = logits.sigmoid()

        log_sigmoid = torch.nn.functional.logsigmoid(logits)
        log_sigmoid_inv = torch.nn.functional.logsigmoid(-logits)

        coef = (target == 1) * self.alpha * torch.pow((self.pos - sigmoid_pred).abs(), self.gamma)
        loss = coef * (self.pos * log_sigmoid + (1 - self.pos) * log_sigmoid_inv)

        coef = (target == 0) * (1 - self.alpha) * torch.pow((sigmoid_pred - self.neg).abs(), self.gamma)
        loss += coef * (self.neg * log_sigmoid + (1 - self.neg) * log_sigmoid_inv)

        return -loss

