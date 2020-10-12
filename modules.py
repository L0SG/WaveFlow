import torch
import torch.nn as nn

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts

@torch.jit.script
def fused_res_skip(tensor, res_skip, n_channels):
    n_channels_int = n_channels[0]
    res = res_skip[:, :n_channels_int]
    skip = res_skip[:, n_channels_int:]
    return (tensor + res), skip

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_h=1, dilation_w=1,
                 causal=True):
        super(Conv2D, self).__init__()
        self.causal = causal
        self.dilation_h, self.dilation_w = dilation_h, dilation_w

        if self.causal:
            self.padding_h = dilation_h * (kernel_size - 1)  # causal along height
        else:
            self.padding_h = dilation_h * (kernel_size - 1) // 2
        self.padding_w = dilation_w * (kernel_size - 1) // 2  # noncausal along width
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              dilation=(dilation_h, dilation_w), padding=(self.padding_h, self.padding_w))
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding_h != 0:
            out = out[:, :, :-self.padding_h, :]
        return out

    def reverse_fast(self, tensor):
        self.conv.padding = (0, self.padding_w)
        out = self.conv(tensor)
        return out


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out

class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size,
                 cin_channels=None, local_conditioning=True, dilation_h=None, dilation_w=None,
                 causal=True):
        super(ResBlock2D, self).__init__()
        self.out_channels = out_channels
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.skip = True
        assert in_channels == out_channels == skip_channels

        self.filter_gate_conv = Conv2D(in_channels, 2*out_channels, kernel_size, dilation_h, dilation_w, causal=causal)

        self.filter_gate_conv_c = nn.Conv2d(cin_channels, 2*out_channels, kernel_size=1)
        self.filter_gate_conv_c = nn.utils.weight_norm(self.filter_gate_conv_c)
        nn.init.kaiming_normal_(self.filter_gate_conv_c.weight)

        self.res_skip_conv = nn.Conv2d(out_channels, 2*in_channels, kernel_size=1)
        self.res_skip_conv = nn.utils.weight_norm(self.res_skip_conv)
        nn.init.kaiming_normal_(self.res_skip_conv.weight)


    def forward(self, tensor, c=None):
        n_channels_tensor = torch.IntTensor([self.out_channels])

        h_filter_gate = self.filter_gate_conv(tensor)
        c_filter_gate = self.filter_gate_conv_c(c)
        out = fused_add_tanh_sigmoid_multiply(h_filter_gate, c_filter_gate, n_channels_tensor)

        res_skip = self.res_skip_conv(out)

        return fused_res_skip(tensor, res_skip, n_channels_tensor)


    def reverse(self, tensor, c=None):
        # used for reverse. c is a cached tensor
        h_filter_gate = self.filter_gate_conv(tensor)
        n_channels_tensor = torch.IntTensor([self.out_channels])
        out = fused_add_tanh_sigmoid_multiply(h_filter_gate, c, n_channels_tensor)

        res_skip = self.res_skip_conv(out)

        return fused_res_skip(tensor, res_skip, n_channels_tensor)

    def reverse_fast(self, tensor, c=None):
        h_filter_gate = self.filter_gate_conv.reverse_fast(tensor)
        n_channels_tensor = torch.IntTensor([self.out_channels])
        out = fused_add_tanh_sigmoid_multiply(h_filter_gate, c, n_channels_tensor)

        res_skip = self.res_skip_conv(out)

        return fused_res_skip(tensor[:, :, -1:, :], res_skip, n_channels_tensor)


class Wavenet2D(nn.Module):
    # a variant of WaveNet-like arch that operates on 2D feature for WF
    def __init__(self, in_channels=1, out_channels=2, num_layers=6,
                 residual_channels=256, gate_channels=256, skip_channels=256,
                 kernel_size=3, cin_channels=80, dilation_h=None, dilation_w=None,
                 causal=True):
        super(Wavenet2D, self).__init__()
        assert dilation_h is not None and dilation_w is not None

        self.residual_channels = residual_channels
        self.skip = True if skip_channels is not None else False

        self.front_conv = nn.Sequential(
            Conv2D(in_channels, residual_channels, 1, 1, 1, causal=causal),
        )

        self.res_blocks = nn.ModuleList()

        for n in range(num_layers):
            self.res_blocks.append(ResBlock2D(residual_channels, gate_channels, skip_channels, kernel_size,
                                              cin_channels=cin_channels, local_conditioning=True,
                                              dilation_h=dilation_h[n], dilation_w=dilation_w[n],
                                              causal=causal))


    def forward(self, x, c=None):
        h = self.front_conv(x)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s

        return skip

    def reverse(self, x, c=None):
        # used for reverse op. c is cached tesnor
        h = self.front_conv(x)    # [B, 64, 1, 13264]
        skip = 0
        for i, f in enumerate(self.res_blocks):
            c_i = c[i]
            h, s = f.reverse(h, c_i) # modification: conv_queue + previous layer's output concat , c_i + conv_queue update: conv_queue last element & previous layer's output concat
            skip += s
        return skip

    def reverse_fast(self, x, c=None):
        # input: [B, 64, 1, T]
        # used for reverse op. c is cached tesnor
        h = self.front_conv(x)  # [B, 64, 1, 13264]
        skip = 0
        for i, f in enumerate(self.res_blocks):
            c_i = c[i]
            h_new = torch.cat((self.conv_queues[i], h), dim=2)  # [B, 64, 3, T]
            h, s = f.reverse_fast(h_new, c_i)
            self.conv_queues[i] = h_new[:, :, 1:, :]  # cache the tensor to queue
            skip += s

        return skip

    def conv_queue_init(self, x):
        self.conv_queues = []
        B, _, _, W = x.size()
        for i in range(len(self.res_blocks)):
            conv_queue = torch.zeros((B, self.residual_channels, 2, W), device=x.device)
            if x.type() == 'torch.cuda.HalfTensor':
                conv_queue = conv_queue.half()
            self.conv_queues.append(conv_queue)