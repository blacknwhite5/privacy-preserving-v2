from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch
import os

from models import Generator, Discriminator, weights_init
from datasets import celebA


# 파라매터
reuse = False
num_epoch = 20
batch_size = 1
lr = 0.00005
alpha = 0.5
beta = 0.999
lambda_photo = 30

# 저장공간생성
if not os.path.exists('images'):
    os.makedirs('images', exist_ok=True)


# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 불러오기
### 이미지 전처리
transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

### celeba 데이터셋
celeba = celebA(path='../../myProject/GANs-pytorch/data/celeba/',
                transforms=transforms)

### 데이터 로드
dataloader = DataLoader(celeba,
                        batch_size=batch_size,
                        shuffle=True)


def main():
    # 네트워크
    G = Generator().to(device)
    D = Discriminator(num_classes=celeba.classes).to(device)

    # weights 초기화
    G.apply(weights_init)
    D.apply(weights_init)

    # pretrained 모델 불러오기
    if reuse:
        assert os.path.isfile('pretrained/ppad_v2.pth')
        checkpoint = torch.load('pretrained/ppad_v2.pth')
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        print('[*]Pretrained model loaded')

    # optimizer 정의
    D_optim = optim.Adam(D.parameters(), lr=lr, betas=(alpha, beta))
    G_optim = optim.Adam(G.parameters(), lr=lr, betas=(alpha, beta))

    # loss 정의
    loss_BCE = nn.BCELoss()
    loss_CE = nn.CrossEntropyLoss()
    loss_L1 = nn.L1Loss()
    loss_MSE = nn.MSELoss()

    for epoch in range(num_epoch):
        for i, (img, cls) in enumerate(dataloader):
            img = img.to(device)
            cls = cls.to(device)

            # # # # #
            # Discriminator
            # # # # #
            fake = G(img)
            D_fake_cls, D_fake = D(fake)
            D_real_cls, D_real = D(img)

            loss_D_real = loss_BCE(D_real, torch.ones_like(D_real))
            loss_D_fake = loss_BCE(D_fake, torch.zeros_like(D_fake))
            loss_D_cls_real = loss_CE(D_real_cls, cls)

            loss_D = loss_D_real + loss_D_fake + loss_D_cls_real

            D_optim.zero_grad()
            loss_D.backward(retain_graph=True)
            D_optim.step()

            # # # # #
            # Generator
            # # # # #
            loss_G_fake = loss_BCE(D_fake, torch.ones_like(D_fake))
            
            ### CAN loss
            loss_G_cls_fake = -((1/celeba.classes)*torch.ones(1, celeba.classes).to(device) \
                                    * nn.LogSoftmax(dim=1)(D_fake_cls)).sum(dim=1).mean()
#            ### MSELoss
#            loss_G_cls_fake = loss_MSE(D_fake_cls, (1/celeba.classes)*torch.ones_like(D_fake_cls))
#
#            ### ppad loss
#            loss_G_cls_fake = 15 - loss_CE(D_fake_cls, cls) 

            loss_photorealistic = loss_L1(img, fake)

            loss_G = loss_G_fake + loss_G_cls_fake + lambda_photo * loss_photorealistic
            
            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()

            # 진행상황 출력
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % \
                  (epoch, num_epoch, i*1, len(celeba), loss_D.item(), loss_G.item()))

            # 이미지 저장
            if i%20000 == 0:
                save_image(torch.cat([img, fake], dim=3), 'images/ep{0:03d}_iter{1:05d}.png'.format(epoch+1, i), normalize=True)

        torch.save({
            'G' : G.state_dict(),
            'D' : D.state_dict()
            },
            'pretrained/ppad_v2.pth')

if __name__ == '__main__':
    main()
