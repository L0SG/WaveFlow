import torch
from torch.nn import functional as F
from math import log, pi

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

def bipartize(x):
    # bipartize the given tensor along height dimension
    # ex: given [H, W] tensor:
    # [0, 4,      [0, 4,
    #  1, 5,       2, 6,
    #  2, 6,       1, 5,
    #  3, 7,] ==>  3, 7,]
    """
    :param x: tensor with shape [B, 1, H, W]
    :return:  same shape with bipartized formulation
    """
    B, _, H, W = x.size()
    assert H % 2 == 0, "height is not even number, bipartize behavior is undefined."
    x_even = x[:, :, ::2, :]
    x_odd = x[:, :, 1::2, :]
    x_out = torch.cat((x_even, x_odd), dim=2)
    return x_out


def unbipartize(x_even, x_odd):
    # reverse op for bipartize
    assert x_even.size() == x_odd.size()
    B, _, H_half, W = x_even.size()
    merged = torch.empty((B, _, H_half*2, W)).to(x_even.device)
    merged[:, :, ::2, :] = x_even
    merged[:, :, 1::2, :] = x_odd

    return merged


def reverse_order(x, dim=2):
    # reverse order of x and c along height dimension
    x = torch.flip(x, dims=(dim,))
    return x


def bipartize_reverse_order(x, dim=2):
    # permutation stragety (b) from waveflow paper
    # ex: given [H, W] tensor:
    # [0, 4,      [1, 5,
    #  1, 5,       0, 4,
    #  2, 6,       3, 7,
    #  3, 7,] ==>  2, 6,]
    """
    :param x: tensor with shape [B, 1, H, W]
    :return:  same shape with permuted height
    """
    B, _, H, W = x.size()
    assert H % 2 == 0, "height is not even number, bipartize behavior is undefined."
    # unsqueeze to (B, _, 1, H, W), reshape to (B, _, 2, H/2, W), then flip on dim with H/2
    x = x.unsqueeze(dim)
    x = x.view(B, _, 2, int(H/2), W)
    x = x.flip(dims=(dim+1,))
    x = x.view(B, _, -1, W)

    return x

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


def apply_affine_coupling_forward(x, log_s, t):
    out = x * torch.exp(log_s) + t
    logdet = torch.sum(log_s)

    return out, logdet


def apply_affine_coupling_inverse(z, log_s, t):
    return ((z - t) * torch.exp(-log_s)).squeeze(2)


# unit test
if __name__ == "__main__":
    test = torch.arange(64).view(8, 8).unsqueeze(0).unsqueeze(0)
    out = bipartize_reverse_order(test)
    out2 = bipartize_reverse_order(out)
    print("")
