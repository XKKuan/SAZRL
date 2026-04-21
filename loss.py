import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvLoss(nn.Module):
    def __init__(self, margin=9.0, temperature=1.0):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, pos_scores, neg_scores, num_neg):
        pos_scores = pos_scores.view(-1, num_neg)
        neg_scores = neg_scores.view(-1, num_neg)

        margin_diff = neg_scores - pos_scores + self.margin
        raw_loss = torch.clamp(margin_diff, min=0)

        score_diff = pos_scores.detach() - neg_scores.detach()
        weights = torch.softmax(score_diff * self.temperature, dim=-1)

        return (weights * raw_loss).sum(dim=-1).mean()


class CRRLoss(nn.Module):
    def __init__(self, t=1.0, p=0.0):
        super().__init__()
        self.t = t
        self.p = p

    def cliff_sigmod(self, x):
        exponent = torch.clamp((self.p - x) / self.t, min=-20, max=20)
        return 1 / (1 + torch.exp(exponent))

    def forward(self, pos_scores, neg_scores, num_neg):
        pos_scores = pos_scores.view(-1, num_neg)
        neg_scores = neg_scores.view(-1, num_neg)
        return (torch.log(torch.sum(self.cliff_sigmod(pos_scores - neg_scores), axis=1) + 1.0)).mean()


class SoftplusLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.criterion = nn.Softplus()
        self.adv_flag = adv_temperature is not None
        if self.adv_flag:
            self.adv_temperature = adv_temperature

    def get_weights(self, n_score):
        return F.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, pos_scores, neg_scores, num_neg):
        pos_scores = pos_scores.view(-1, num_neg)
        neg_scores = neg_scores.view(-1, num_neg)

        if self.adv_flag:
            return (self.criterion(-pos_scores).mean() + (self.get_weights(neg_scores) * self.criterion(neg_scores)).sum(dim=-1).mean()) / 2
        else:
            return (self.criterion(-pos_scores).mean() + self.criterion(neg_scores).mean()) / 2


class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super(SigmoidLoss, self).__init__()
        self.criterion = nn.LogSigmoid()
        self.adv_flag = adv_temperature is not None
        if self.adv_flag:
            self.adv_temperature = adv_temperature

    def get_weights(self, n_score):
        return F.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, pos_scores, neg_scores, num_neg):
        pos_scores = pos_scores.view(-1, num_neg)
        neg_scores = neg_scores.view(-1, num_neg)

        if self.adv_flag:
            return -(self.criterion(pos_scores).mean() + (self.get_weights(neg_scores) * self.criterion(-neg_scores)).sum(dim=-1).mean()) / 2
        else:
            return -(self.criterion(pos_scores).mean() + self.criterion(-neg_scores).mean()) / 2
