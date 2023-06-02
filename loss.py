import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, z1, z2):
        # z1: (B, F), z2: (B, F)
        # TODO: One possible issue is that the values of features depend on their size. It is necessary to take into scaling using their size.
        z1 = F.normalize(z1, p=2.0, dim=-1)
        z2 = F.normalize(z2, p=2.0, dim=-1)

        z1z2 = torch.cat((z1, z2), 0)  # (2B, F)
        S = torch.matmul(z1z2, z1z2.t())  # (2B, 2B)
        S = S / self.tau
        S = torch.exp(S)
        I = 1 - torch.eye(S.size(0), device=S.device).int()
        S = S * I
        # logging.info("S: {}".format(S.detach().cpu().numpy()))  #
        B = z1.size(0)
        i = torch.arange(B)
        n = torch.cat((S[i, i+B], S[i+B, i]))
        d = torch.sum(S, dim=-1)
        # logging.info("n: {}".format(n.detach().cpu().numpy()))  #
        # logging.info("d: {}".format(d.detach().cpu().numpy()))  #
        l = -torch.log(torch.true_divide(n, d))
        # logging.info("l: {}".format(l.detach().cpu().numpy()))  #
        l = torch.sum(l) / (2 * B)
        return l

class ContrastiveLoss_2(nn.Module):
    def __init__(self, tau):
        super(ContrastiveLoss_2, self).__init__()
        self.tau = tau

    def forward(self, z_1, z_2):
        # z_1, z_2: [B, F] => [B, 1, F]
        z_1 = torch.unsqueeze(z_1, dim=1)
        z_2 = torch.unsqueeze(z_2, dim=1)
        B, T = z_1.size(0), z_1.size(1)
        z = torch.cat((z_1, z_2), dim=0)  # [2B, 1, F]
        z = z.transpose(0, 1)  # [1, 2B, F]
        sim = torch.matmul(z, z.transpose(1, 2))  # [1, 2B, 2B]
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits /= self.tau
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z_1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss
