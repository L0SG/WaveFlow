import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_h=1, dilation_w=1):
        super(Conv2D, self).__init__()
        self.padding_h = dilation_h * (kernel_size - 1) # causal along height
        self.padding_w = dilation_w * (kernel_size - 1) // 2  # noncausal along width
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              dilation=(dilation_h, dilation_w), padding=(self.padding_h, self.padding_w))
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.padding_h != 0:
            out = out[:, :, :-self.padding_h, :]
        return out


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))

    def forward(self, x):
        out = self.conv(x)
        out = out * torch.exp(self.scale * 3)
        return out


class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size,
                 cin_channels=None, local_conditioning=True, dilation_h=None, dilation_w=None):
        super(ResBlock2D, self).__init__()
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.skip = True if skip_channels is not None else False

        self.filter_conv = Conv2D(in_channels, out_channels, kernel_size, dilation_h, dilation_w)
        self.gate_conv = Conv2D(in_channels, out_channels, kernel_size, dilation_h, dilation_w)
        self.res_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        if self.skip:
            self.skip_conv = nn.Conv2d(out_channels, skip_channels, kernel_size=1)
            self.skip_conv = nn.utils.weight_norm(self.skip_conv)
            nn.init.kaiming_normal_(self.skip_conv.weight)

        if self.local_conditioning:
            self.filter_conv_c = nn.Conv2d(cin_channels, out_channels, kernel_size=1)
            self.gate_conv_c = nn.Conv2d(cin_channels, out_channels, kernel_size=1)
            self.filter_conv_c = nn.utils.weight_norm(self.filter_conv_c)
            self.gate_conv_c = nn.utils.weight_norm(self.gate_conv_c)
            nn.init.kaiming_normal_(self.filter_conv_c.weight)
            nn.init.kaiming_normal_(self.gate_conv_c.weight)

    def forward(self, tensor, c=None):
        h_filter = self.filter_conv(tensor)
        h_gate = self.gate_conv(tensor)

        if self.local_conditioning:
            h_filter += self.filter_conv_c(c)
            h_gate += self.gate_conv_c(c)

        out = torch.tanh(h_filter) * torch.sigmoid(h_gate)

        res = self.res_conv(out)
        skip = self.skip_conv(out) if self.skip else None
        return (tensor + res) * math.sqrt(0.5), skip


class Wavenet2D(nn.Module):
    # a varient of WaveNet-like arch that operates on 2D feature for WF
    def __init__(self, in_channels=1, out_channels=2, num_layers=6,
                 residual_channels=256, gate_channels=256, skip_channels=256,
                 kernel_size=3, cin_channels=80, dilation_h=None, dilation_w=None):
        super(Wavenet2D, self).__init__()
        assert dilation_h is not None and dilation_w is not None

        self.skip = True if skip_channels is not None else False

        self.front_conv = nn.Sequential(
            Conv2D(in_channels, residual_channels, 1, 1, 1),
        )

        self.res_blocks = nn.ModuleList()
        for n in range(num_layers):
             self.res_blocks.append(ResBlock2D(residual_channels, gate_channels, skip_channels,
                                                    kernel_size,
                                                    cin_channels=cin_channels, local_conditioning=True,
                                                    dilation_h=dilation_h[n], dilation_w=dilation_w[n]))

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            Conv2D(residual_channels, out_channels, 1, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x, c=None):
        h = self.front_conv(x)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            if self.skip:
                h, s = f(h, c)
                skip += s
            else:
                h, _ = f(h, c)
        if self.skip:
            out = self.final_conv(skip)
        else:
            out = self.final_conv(h)

        return out