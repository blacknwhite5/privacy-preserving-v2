import torch.nn as nn
import torch


class resnet_9blocks(nn.Module):
    def __init__(self, in_channel=4, out_channel=3, last_activation='tanh'):
        super(resnet_9blocks, self).__init__()
        
        self.down_sample = nn.Sequential(nn.ReflectionPad2d(in_channel),
                                         DownBlock(in_channel, 64, 7, 1, 0),
                                         DownBlock(64, 128, 3, 2, 1),
                                         DownBlock(128, 256, 3, 2, 1))

        self.resnet_blocks = []
        for _ in range(9):
            self.resnet_blocks += [ResnetBlock(256, 256, 3, 1)]
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

#        up_blocks = [UpBlock(256, 128, 3, 2, 1),
#                     UpBlock(128, 64, 3, 2, 1),
#                     nn.ReflectionPad2d(3),
#                     nn.Conv2d(64, 3, kernel_size=7, stride=1)]
#
#        if last_activation == 'tanh':
#            up_blocks.append(nn.Tanh())
#        elif last_activation == 'none':
#            pass
#        else:
#            raise NotImplementedError("should be 'tanh' or 'none'")
#
#        self.up_sample = nn.Sequential(*up_blocks)
        
        self.pad = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=7, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    # TODO: landmark, layernorm
    def forward(self, x, face_info):
        x = torch.cat([x, face_info], dim=1)
        x = self.down_sample(x)
        x = self.resnet_blocks(x)
#        out = self.up_sample(x)
        x = nn.functional.interpolate(x, (64, 64))
        x = self.conv1(x)
        x = self.relu(x)
        x = nn.functional.interpolate(x, (128, 128))
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pad(x)
        x = self.conv3(x)
        out = self.tanh(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, activation='relu', norm=True):
        super(UpBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, 
                                     padding=padding, output_padding=padding)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        self.upblock = nn.Sequential(*layers) 

    def forward(self, x):
        out = self.upblock(x)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, activation='relu', norm=True):
        super(DownBlock, self).__init__()
        self.downblock = self._make_downblock(in_channel, out_channel, kernel_size, stride, padding, activation, norm)

    def _make_downblock(self, in_channel, out_channel, kernel_size, stride, padding, activation, norm):
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers) 

    def forward(self, x):
        return self.downblock(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=4, padding=1, activation='relu', norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(in_channel, out_channel, kernel_size, stride, padding, activation, norm)

    def build_conv_block(self, in_channel, out_channel,
                         kernel_size=4, stride=1, padding=1, activation='relu', norm=True):
        layers = []
        for i in range(2):
            layers += [nn.ReflectionPad2d(padding),
                       nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_channel))
            if activation == 'relu' and i != 1:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# TODO : need to modify
class PatchGAN(nn.Module):
    def __init__(self, num_classes, last_activation='sigmoid'):
        super(PatchGAN, self).__init__()
        self.num_classes = num_classes
        self.last_activation = last_activation

        def block(layer, in_channel, out_channel,
                  kernel_size=4, stride=2, padding=1, activation='lrelu', norm=True):
            layers = []
            # layer
            if layer == 'conv':
                layers.append(nn.Conv2d(in_channel, out_channel,
                                        kernel_size=kernel_size, stride=stride, padding=padding))
            elif layer == 'linear':
                layers.append(nn.Linear(in_channel, out_channel))
            else:
                raise NotImplementedError('Illegal layer, opts: conv, linear')

            # norm
            if norm:
                layers.append(nn.BatchNorm2d(out_channel))

            # activation
            if activation == 'lrelu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'softmax':
                layers.append(nn.Softmax(dim=0))
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'none':
                pass
            else:
                raise NotImplementedError('Illegal activation, opts: lrelu, softmax, sigmoid, none')
            return layers


        self.feature = nn.Sequential(*block('conv', 3, 32, norm=False),
                                     *block('conv', 32, 64),
                                     *block('conv', 64, 128),
                                     *block('conv', 128, 256),
                                     *block('conv', 256, 512),
                                     *block('conv', 512, 512))

        self.classifier = nn.Sequential(*block('linear', 512*2*2, 1024, norm=False),
                                        *block('linear', 1024, 512, norm=False),
                                        *block('linear', 512, self.num_classes, norm=False, activation='none'))

        self.discriminator = nn.Sequential(*block('linear', 512*2*2, 1, norm=False, activation=self.last_activation))

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        real_or_fake = self.discriminator(out)
        cls = self.classifier(out)
        return cls, real_or_fake


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    import torch
    inputs = torch.Tensor(16,3,128,128)
    G = Generator()
    print(G)
    G = G.to('cuda')
    inputs = inputs.to('cuda')
    print(G(inputs).shape)
