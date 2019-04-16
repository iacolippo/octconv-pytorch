import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OctConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, alphas=(0.5, 0.5)):
        super(OctConv, self).__init__()
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_in <= 1, "Alphas must be in interval [0, 1]"

        # CH IN
        self.ch_in_hf = int((1 - self.alpha_in) * ch_in)
        self.ch_in_lf = ch_in - self.ch_in_hf

        # CH OUT
        self.ch_out_hf = int((1 - self.alpha_out) * ch_out)
        self.ch_out_lf = ch_out - self.ch_out_hf

        # FILTERS
        self.wHtoH = nn.Parameter(torch.randn(self.ch_out_hf, self.ch_in_hf, kernel_size, kernel_size))
        self.wHtoL = nn.Parameter(torch.randn(self.ch_out_lf, self.ch_in_hf, kernel_size, kernel_size))
        self.wLtoH = nn.Parameter(torch.randn(self.ch_out_hf, self.ch_in_lf, kernel_size, kernel_size))
        self.wLtoL = nn.Parameter(torch.randn(self.ch_out_lf, self.ch_in_lf, kernel_size, kernel_size))

        # PADDING: (H - F + 2P)/S + 1 = 2 * [(0.5 H - F + 2P)/S +1] -> P = (F-S)/2
        self.padding = (kernel_size - stride) // 2

    def forward(self, input):
        # logic to handle input tensors:
        # if alpha_in = 0., we assume to be at the first layer, with only high freq repr
        if self.alpha_in == 0:
            hf_input = input
            lf_input = torch.Tensor([]).reshape(1, 0)
        else:
            fmap_size = input.shape[-1]
            hf_input = input[:, :self.ch_in_hf * 4, ...].reshape(-1, self.ch_in_hf, fmap_size * 2, fmap_size * 2)
            lf_input = input[:, self.ch_in_hf * 4:, ...]

        HtoH = HtoL = LtoL = LtoH = 0.
        if self.alpha_in < 1:
            # if alpha < 1 there is high freq component
            if self.ch_out_hf > 0:
                HtoH = F.conv2d(hf_input, self.wHtoH, padding=self.padding)
            if self.ch_out_lf > 0:
                HtoL = F.conv2d(F.avg_pool2d(hf_input, 2), self.wHtoL, padding=self.padding)
        if self.alpha_in > 0:
            # if alpha > 0 there is low freq component
            if self.ch_out_hf > 0:
                LtoH = F.interpolate(F.conv2d(lf_input, self.wLtoH, padding=self.padding),
                                     scale_factor=2, mode='nearest')
            if self.ch_out_lf > 0:
                LtoL = F.conv2d(lf_input, self.wLtoL, padding=self.padding)

        hf_output = HtoH + LtoH
        lf_output = LtoL + HtoL
        if 0 < self.alpha_out < 1:
            # if alpha in (0, 1)
            fmap_size = hf_output.shape[-1] // 2
            hf_output = hf_output.reshape(-1, 4 * self.ch_out_hf, fmap_size, fmap_size)
            output = torch.cat([hf_output, lf_output], dim=1)  # cat over channel dim
        elif np.isclose(self.alpha_out, 1., atol=1e-8):
            # if only low req (alpha_out = 1.)
            output = lf_output
        elif np.isclose(self.alpha_out, 0., atol=1e-8):
            # if only high freq (alpha_out = 0.)
            output = hf_output
        return output


oc = OctConv(ch_in=3, ch_out=3, kernel_size=3, alphas=(0., 0.5))
oc1 = OctConv(ch_in=3, ch_out=10, kernel_size=7, alphas=(0.5, 0.8))
oc2 = OctConv(ch_in=10, ch_out=1, kernel_size=3, alphas=(0.8, 0.))
out = oc2(oc1(oc(torch.randn(2, 3, 32, 32))))
print(out.shape)
