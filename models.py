import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def UpDownBlock(mode, in_channel, out_channel,
                        kernel_size, stride, padding, activation='relu', norm=True):
            if mode == 'down':
                layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
            elif mode == 'up':
                layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, 
                                             padding=padding, output_padding=padding)]


            if norm:
                layers.append(nn.InstanceNorm2d(out_channel))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            return layers

        def ResnetBlock(in_channel, out_channel,
                        kernel_size=4, stride=1, padding=1, activation='relu', norm=True):
            layers = []
            for i in range(2):
                layers.extend([nn.ReflectionPad2d(padding),
                               nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)])
                if norm:
                    layers.append(nn.InstanceNorm2d(out_channel))
                if activation == 'relu' and i != 1:
                    layers.append(nn.ReLU(inplace=True))
            return layers
        
        self.down_sample = nn.Sequential(nn.ReflectionPad2d(3),
                                         *UpDownBlock('down', 3, 64, 7, 1, 0),
                                         *UpDownBlock('down', 64, 128, 3, 2, 1),
                                         *UpDownBlock('down', 128, 256, 3, 2, 1))

        b1 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b2 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b3 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b4 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b5 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b6 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b7 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b8 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        b9 = nn.Sequential(*ResnetBlock(256, 256, 3, 1))
        self.resnet_blocks = nn.Sequential(b1, b2, b3, b4, b5, b6, b7, b8, b9)

        self.up_sample = nn.Sequential(*UpDownBlock('up', 256, 128, 3, 2, 1),
                                       *UpDownBlock('up', 128, 64, 3, 2, 1),
                                       nn.ReflectionPad2d(3),
                                       nn.Conv2d(64, 3, kernel_size=9, stride=1),
                                       nn.Tanh())

    def forward(self, x, pose):
        x = self.down_sample(x)
        x = self.resnet_blocks(x)
        out = self.up_sample(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

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

        self.classifier = nn.Sequential(*block('linear', 512*3*2, 1024, norm=False),
                                        *block('linear', 1024, 512, norm=False),
                                        *block('linear', 512, self.num_classes, norm=False, activation='none'))

#        self.discriminator = nn.Sequential(*block('linear', 512*3*2, 1, norm=False, activation='sigmoid'))
        self.discriminator = nn.Sequential(*block('linear', 512*3*2, 1, norm=False, activation='none'))

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
