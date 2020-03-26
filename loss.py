import torch

class GANLoss:
    def __init__(self, reduction='sum'):
        self.criterion = torch.nn.BCELoss()
        self.reduction = reduction

    def __call__(self, *args):
        if len(args) == 2:
            real, fake = args
            real_loss = self.criterion(real, torch.ones_like(real))
            fake_loss = self.criterion(fake, torch.zeros_like(fake))
            if self.reduction == 'sum':
                loss = real_loss + fake_loss
            elif self.reduction == 'mean':
                loss = 0.5 * (real_loss + fake_loss)
            else:
                raise NotImplementedError(
                f"reduction was input: {self.reduction}, only available 'sum' and 'mean'")
        elif len(args) == 1:
            fake = args[0]
            loss = self.criterion(fake, torch.ones_like(fake))
        return loss


class LSGANLoss(GANLoss):
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()


class WGAN_GPLoss:
    def __init__(self, D, lambda_gp):
        self.D = D
        self.lambda_gp = lambda_gp

    def __call__(self, *args):
        if len(args) == 2:
            real, fake = args
            d_real = self.D(real)
            d_fake = self.D(fake)
            if isinstance(d_real, tuple):
                d_real = d_real[1]
                d_fake = d_fake[1]
            gradient_penalty = self.gradient_penalty(real, fake)
            return -torch.mean(d_real) + torch.mean(d_fake) + self.lambda_gp * gradient_penalty
        elif len(args) == 1:
            fake = args[0]
            d_fake = self.D(fake)
            if isinstance(d_fake, tuple):
                d_fake = d_fake[1]
            return -torch.mean(d_fake)
        else:
            raise Exception(f'inputs must be 1 or 2, but got {len(args)}')

    def gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        eps = torch.rand((real_samples.size(0), 1, 1, 1)).to(real_samples.device)
        x_hat = (eps * real_samples.data + (1-eps) * fake_samples.data).requires_grad_(True)
        d_hat = self.D(x_hat)
        if isinstance(d_hat, tuple):
            d_hat = d_hat[1]
        ones = torch.ones((real_samples.shape[0], 1), requires_grad=False).to(real_samples.device)
        gradients = torch.autograd.grad(outputs=d_hat,
                                        inputs=x_hat,
                                        grad_outputs=ones,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
        return gradient_penalty


class CANLoss:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, D_fake_cls):
        return -((1/self.num_classes)*torch.ones(1, self.num_classes).to(D_fake_cls.device) \
                * torch.nn.LogSoftmax(dim=1)(D_fake_cls)).sum(dim=1).mean()

class CELoss:
    def __init__(self):
        self.celoss = torch.nn.CrossEntropyLoss()

    def __call__(self, real_cls, fake_cls):
        return self.celoss(real_cls, fake_cls)

class PixelwiseLoss:
    def __init__(self, lambda_photo=100):
        self.l1loss = torch.nn.L1Loss()
        self.lambda_photo = lambda_photo

    def __call__(self, real, fake):
        return self.lambda_photo * self.l1loss(real, fake)

    def __repr__(self):
        return "L1loss between real image and fake image. default lambda is 100"

# TODO: need to test
class PPADLoss:
    """https://github.com/jason718/PPAD/blob/master/tools/train_net.py"""
    def __init__(self):
        self.celoss = torch.nn.CrossEntropyLoss()

    def __call__(self, real_cls, fake_cls):
        return 15 - self.celoss(real_cls, fake_cls)


if __name__ == '__main__':
    from models import Discriminator
    D = Discriminator(num_classes=1)
    a = torch.ones([16, 3, 218, 178])
    b = torch.zeros([16, 3, 218, 178])
    criterion = WGAN_GPLoss(D, 10)
    print(criterion(a,b))
    print(criterion(b))

    crit2 = GANLoss()
    print(crit2(a,b))
    
    crit3 = LSGANLoss()
    print(crit3(a,b))

    crit4 = pixelwiseloss()
    print(crit4(a,b))

    num_cls = 10
    crit5 = CANLoss(num_cls)
    print(crit5(torch.Tensor(num_cls, 10)))
