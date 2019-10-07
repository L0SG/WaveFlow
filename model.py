import torch
from torch import nn
from math import log, pi
from modules import Wavenet2D, Conv2D, ZeroConv2d
import math
import torch.nn.functional as F
from torch.distributions.normal import Normal

logabs = lambda x: torch.log(torch.abs(x))


class WaveFlowCoupling2D(nn.Module):
    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=6, num_height=None, layers_per_dilation_h_cycle=3):
        super().__init__()
        assert num_height is not None
        self.num_height = num_height
        self.layers_per_dilation_h_cycle = layers_per_dilation_h_cycle
        # dilation for width & height generation loop
        self.dilation_h = []
        self.dilation_w = []
        self.kernel_size = 3
        for i in range(num_layer):
            self.dilation_h.append(2**(i % self.layers_per_dilation_h_cycle))
            self.dilation_w.append(2**i)

        self.net = Wavenet2D(in_channels=in_channel, out_channels=filter_size,
                           num_layers=num_layer, residual_channels=filter_size,
                           gate_channels=filter_size, skip_channels=filter_size,
                           kernel_size=3, cin_channels=cin_channel, dilation_h=self.dilation_h, dilation_w=self.dilation_w)
        # projector for output, log_s and t
        self.proj_log_s = ZeroConv2d(filter_size, in_channel)
        self.proj_t = ZeroConv2d(filter_size, in_channel)

    def forward(self, x, c=None):
        x_shift = shift_1d(x)

        feat = self.net(x_shift, c)
        log_s = self.proj_log_s(feat)
        t = self.proj_t(feat)

        out = x * torch.exp(log_s) + t
        logdet = torch.sum(log_s)
        return out, logdet

    def reverse(self, z, c=None, x=None):
        x_shift = shift_1d(x)

        for i_h in range(self.num_height):
            x_in, c_in = x_shift[:, :, :i_h+1, :], c[:, :, :i_h+1, :]
            feat = self.net(x_in, c_in)[:, :, -1, :].unsqueeze(2)
            log_s = self.proj_log_s(feat)
            t = self.proj_t(feat)

            z_trans = z[:, :, i_h, :].unsqueeze(2)
            z[:, :, i_h, :] = (z_trans - t) * torch.exp(-log_s)
            x[:, :, i_h, :] = z[:, :, i_h, :]
            if i_h != (self.num_height -1):
                x_shift[:, :, i_h+1] = x[:, :, i_h, :]
        return z, c, x



def reverse_order(x):
    # reverse order of x and c along channel dimension (instead of change_order of bipartite flow)
    x = torch.flip(x, dims=(2,))
    return x


class Flow(nn.Module):
    def __init__(self, in_channel, cin_channel, filter_size, num_layer, num_height, layers_per_dilation_h_cycle):
        super().__init__()

        self.coupling = WaveFlowCoupling2D(in_channel, cin_channel, filter_size=filter_size, num_layer=num_layer, num_height=num_height,
                                           layers_per_dilation_h_cycle=layers_per_dilation_h_cycle,)

    def forward(self, x, c=None):
        out, logdet = self.coupling(x, c)
        out = reverse_order(out)
        c = reverse_order(c)

        return out, c, logdet

    def reverse(self, z, c=None, x=None):
        z = reverse_order(z)
        c = reverse_order(c)
        x = reverse_order(x)
        z, c, x = self.coupling.reverse(z, c, x)
        return z, c, x


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


def squeeze_to_2d(x, c, h):
    if x is not None:  # during synthesize phase, we feed x as None
        # squeeze 1D waveform x into 2d matrix given height h
        B, C, T = x.size()
        assert T % h == 0, "cannot make 2D matrix of size {} given h={}".format(T, h)
        x = x.view(B, int(T / h), C * h)
        # permute to make column-major 2D matrix of waveform
        x = x.permute(0, 2, 1)
        # unsqueeze to have tensor shape of [B, 1, H, W]
        x = x.unsqueeze(1)

    # same goes to c, but keeping the 2D mel-spec shape
    B, C, T = c.size()
    c = c.view(B, C, int(T / h), h)
    c = c.permute(0, 1, 3, 2)

    return x, c


def unsqueeze_to_1d(x, h):
    # unsqueeze 2d tensor back to 1d representation
    B, C, H, W = x.size()
    assert H == h, "wrong height given, must match model's n_height {} and given tensor height {}.".format(h, H)
    x = x.permute(0, 1, 3, 2)
    x = x.contiguous().view(B, C, -1)
    x = x.squeeze(1)

    return x


def shift_1d(x):
    # shift tensor on height by one for WaveFlowAR modeling
    x = F.pad(x, (0, 0, 1, 0))
    x = x[:, :, :-1, :]
    return x


class WaveFlow(nn.Module):
    def __init__(self, in_channel, cin_channel, res_channel, n_height, n_flow, n_layer, layers_per_dilation_h_cycle):
        super().__init__()
        self.in_channel = in_channel
        self.cin_channel = cin_channel
        self.res_channel = res_channel
        self.n_height = n_height
        self.n_flow = n_flow
        self.n_layer = n_layer

        self.layers_per_dilation_h_cycle = layers_per_dilation_h_cycle

        self.flows = nn.ModuleList()
        for i in range(self.n_flow):
            self.flows.append(Flow(self.in_channel, self.cin_channel, filter_size=self.res_channel,
                                   num_layer=self.n_layer, num_height=self.n_height,
                                   layers_per_dilation_h_cycle=self.layers_per_dilation_h_cycle,
                                   ))

        self.upsample_conv = nn.ModuleList()
        for s in [16, 16]:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

    def forward(self, x, c):
        B, _, T = x.size()
        logdet, log_p_sum = 0, 0

        c = self.upsample(c)
        x, c = squeeze_to_2d(x, c, h=self.n_height)
        out = x

        for flow in self.flows:
            out, c, logdet_new = flow(out, c)
            logdet = logdet + logdet_new
        log_p_sum += ((-0.5) * (log(2.0 * pi) + 2.0 * out.pow(2)).sum())

        logdet = logdet / (B * T)
        log_p = log_p_sum / (B * T)

        return log_p, logdet

    def reverse(self, c, temp=1.0):

        c = self.upsample(c)
        B, _, T_c = c.size()
        len_pad = T_c % self.n_height
        # right zero-pad upsampled mel-spec
        c = F.pad(c, (0, len_pad))
        _, c = squeeze_to_2d(None, c, h=self.n_height)

        # sample gaussian noise that matches c
        q_0 = Normal(c.new_zeros((B, 1, c.size()[2], c.size()[3])), c.new_ones((B, 1, c.size()[2], c.size()[3])))
        z = q_0.sample() * temp

        # container that stores the generated output
        x = c.new_zeros((B, 1, c.size()[2], c.size()[3]))

        for i, flow in enumerate(self.flows[::-1]):
            z, c, x = flow.reverse(z, c, x)

        x = unsqueeze_to_1d(x, self.n_height)
        if len_pad != 0:
            x = x[:, :-len_pad]
        return x

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c



if __name__ == "__main__":
    # x = torch.arange(end=7*15872).view((7, 1, 15872)).float().cuda()
    # c = torch.arange(end=7*80*62).view((7, 80, 62)).float().cuda()
    # net = WaveFlow(1, 80, 128, 64, 8, 8, 5).cuda()
    # out = net(x, c)

    #x = torch.arange(end=1*15872).view((1, 1, 15872)).float().cuda()
    c = torch.arange(end=1*80*65).view((1, 80, 65)).float().cuda()
    net = WaveFlow(1, 80, 128, 64, 8, 8, 5).cuda().eval()
    with torch.no_grad():
        out = net.reverse(c)