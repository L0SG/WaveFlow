import torch
from math import log, pi

# class WaveFlowLoss(torch.nn.Module):
#     def __init__(self, sigma=1.0):
#         super(WaveFlowLoss, self).__init__()
#         self.sigma = sigma
#
#     def forward(self, model_output):
#         out, logdet = model_output
#         B, _, C, T = out.size()
#         loss = (0.5) * (log(2.0 * pi) + 2 * log(self.sigma) + out.pow(2) / (self.sigma*self.sigma)).sum() - logdet
#         return loss / (B*C*T)

class WaveFlowLossDataParallel(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveFlowLossDataParallel, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        out, logdet = model_output
        logdet = logdet.sum()
        B, _, C, T = out.size()
        loss = (0.5) * (log(2.0 * pi) + 2 * log(self.sigma) + out.pow(2) / (self.sigma*self.sigma)).sum() - logdet
        return loss / (B*C*T)