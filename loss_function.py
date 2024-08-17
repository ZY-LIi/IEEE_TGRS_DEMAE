import torch
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, n_class, alpha=1, gamma=0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.Tensor([alpha] * n_class)
        self.gamma = gamma

    def forward(self, y_hat, y):
        self.alpha = self.alpha.to(y_hat.device)
        pred_sm = y_hat.softmax(dim=-1)
        pt = pred_sm.gather(dim=1, index=y.view(-1, 1))
        alpha = self.alpha.gather(dim=-1, index=y)
        loss = -alpha * (1 - pt)**self.gamma * pt.log()
        return loss.mean()


class SNREnhanced_FocalLoss(nn.Module):
    def __init__(self, scale=1.0):
        super(SNREnhanced_FocalLoss, self).__init__()
        self.scale = scale

    def forward(self, y_hat, y, snr):
        pred_sm = y_hat.softmax(dim=-1)
        pt = pred_sm.gather(dim=1, index=y.view(-1, 1))
        snr = snr.to(y.device)
        loss = -1 * (1 - pt) ** (2 * torch.sigmoid(-snr * self.scale)).unsqueeze(1) * pt.log()
        return loss.mean()


class SNREnhanced_MSELoss(nn.Module):
    def __init__(self, scale=1.2):
        super(SNREnhanced_MSELoss, self).__init__()
        self.scale = scale
        self.loss_func = nn.MSELoss(reduction='none')

    def forward(self, x_hat, x, snr):
        B, N, C = x.shape
        loss = self.loss_func(x_hat, x)
        loss = loss.reshape(B, N * C)
        snr = snr.to(x.device)
        loss = loss.mean(dim=1) * torch.exp(-snr)
        return loss.mean() * self.scale


class nllloss(nn.Module):
    def __init__(self):
        super(nllloss, self).__init__()
        self.loss_func = nn.NLLLoss()

    def forward(self, y_hat, y):
        return self.loss_func(y_hat, y)

