import torch.nn as nn
import torch

resl_to_ch = {
    4    : (512, 512),
    8    : (512, 512),
    16   : (512, 512),
    32   : (512, 512),
    64   : (512, 256),
    128  : (256, 128),
    256  : (128, 64),
    512  : (64, 32),
    1024 : (32, 16),
}

resl_to_batch = {
    4    : 512,
    8    : 512,
    16   : 512,
    32   : 256,
    64   : 128,
    128  : 64,
    256  : 32,
    512  : 16,
    1024 : 12
}

resl_to_lr = {
    4    : 0.001,
    8    : 0.001,
    16   : 0.001,
    32   : 0.001,
    64   : 0.0015,
    128  : 0.0015,
    256  : 0.002,
    512  : 0.003,
    1024 : 0.003
}

class EqualizedLR(nn.Module):
    def __init__(self, layer):
        super().__init__()

        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        layer.bias.data.fill_(0)

        self.wscale = layer.weight.data.detach().pow(2.).mean().sqrt()
        layer.weight.data /= self.wscale

        self.layer = layer

    def forward(self, x):
        return self.layer(x * self.wscale)


class PixelWiseNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_square_mean = x.pow(2).mean(dim=1, keepdim=True)
        denom = torch.rsqrt(x_square_mean + 1e-8)
        return x * denom


class ToRGBLayer(nn.Module):
    def __init__(self, resl, rgb_channel):
        super().__init__()
        _, in_c  = resl_to_ch[resl]

        self.conv = nn.Sequential(
            EqualizedLR(nn.Conv2d(in_c, rgb_channel, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class ReslBlock(nn.Module):
    def __init__(self, resl):
        super().__init__()
        in_c, out_c  = resl_to_ch[resl]

        self.conv = nn.Sequential(
            EqualizedLR(nn.Conv2d(in_c, out_c, 3, 1, 1)),
            PixelWiseNorm(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualizedLR(nn.Conv2d(out_c, out_c, 3, 1, 1)),
            PixelWiseNorm(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        up   = F.interpolate(x, scale_factor=2)
        conv = self.conv(up)
        return conv


class G(nn.Module):
    def __init__(self, resl=4, rgb_channel=3):
        super().__init__()
        self.resl = resl
        self.rgb_channel = rgb_channel


        in_c, out_c = resl_to_ch[resl]

        self.resl_blocks = nn.Sequential(
            # Original Repo using "tf.nn.conv2d_transpose"
            EqualizedLR(nn.ConvTranspose2d(in_c, out_c, 4)),
            # nn.Upsample(size=(4, 4)),
            # EqualizedLR(nn.Conv2d(in_c, out_c, 1, 1, 0)),
            PixelWiseNorm(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualizedLR(nn.Conv2d(out_c, out_c, 3, 1, 1)),
            PixelWiseNorm(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.rgb_l = None
        self.rgb_h = ToRGBLayer(self.resl, self.rgb_channel)
        self.alpha = 0

    def forward(self, x, phase):
        if phase == "transition":
            return self.transition_forward(x)
        elif phase == "stabilization":
            return self.stabilization_forward(x)

    def grow_network(self):
        self.resl *= 2
        self.resl_blocks = nn.Sequential(*self.resl_blocks, ReslBlock(self.resl))
        self.rgb_l = self.rgb_h
        self.rgb_h = ToRGBLayer(self.resl, self.rgb_channel)
        self.alpha = 0

    def transition_forward(self, x):
        x = self.resl_blocks[:-1](x)

        # experiment candidate : apply rgb_l first and succeeding interpolate
        # low resolution path
        x_up = F.interpolate(x, scale_factor=2)
        rgb_l = self.rgb_l(x_up)

        # high resolution path
        x = self.resl_blocks[-1](x)
        rgb_h = self.rgb_h(x)

        return self.alpha * (rgb_h - rgb_l) + rgb_l

    def stabilization_forward(self, x):
        x = self.resl_blocks(x)
        rgb_h = self.rgb_h(x)
        return rgb_h

    def update_alpha(self, delta):
        self.alpha += delta
        self.alpha = min(1, self.alpha)
        

if __name__ == '__main__':
    gg = G()
    print(gg)
    print('-'*100)

#    gg.grow_network()
#    print(gg)
#    print('-'*100)
#
#    gg.grow_network()
#    print(gg)
#    print('-'*100)
