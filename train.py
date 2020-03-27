from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
import argparse
import os

from models import Generator, Discriminator, weights_init
from celeba_aligned import celebA
from utils import *
from loss import *


parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=int, default=0.0002)
parser.add_argument('--b1', type=int, default=0.5)
parser.add_argument('--b2', type=int, default=0.999)
parser.add_argument('--lambda_photo', type=int, default=100)
parser.add_argument('--lambda_gp', type=int, default=10)
parser.add_argument('--n_repeat', type=int, default=5)
parser.add_argument('--reuse', dest='reuse', action='store_true')
parser.set_defaults(norm=False)
opts = parser.parse_args()
print(opts)

# seed 고정
torch.manual_seed(0)

# 저장공간생성
os.makedirs('images', exist_ok=True)
os.makedirs('pretrained', exist_ok=True)


# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 불러오기
### 이미지 전처리
transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

### celeba 데이터셋
celeba = celebA(path='/home/moohyun/Desktop/myProject/GANs-pytorch/data/celeba/',
                transforms=transforms)

### 데이터 로드
dataloader = DataLoader(celeba,
                        batch_size=opts.batch_size,
                        shuffle=True)

def main():
    # 네트워크
    G = Generator().to(device)
    D = Discriminator(num_classes=celeba.classes, last_activation='none').to(device)

#    parallel = True
#    if parallel:
#        G = nn.DataParallel(G)
#        D = nn.DataParallel(D)
#    print(G)

    # weights 초기화
    G.apply(weights_init)
    D.apply(weights_init)

    # pretrained 모델 불러오기
    if opts.reuse:
        assert os.path.isfile('pretrained/ppad_v2.pth')
        checkpoint = torch.load('pretrained/ppad_v2.pth')
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        print('[*]Pretrained model loaded')

    # optimizer 정의
    D_optim = optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))
    G_optim = optim.Adam(G.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))

    # loss 정의
    wganloss = WGAN_GPLoss(D, opts.lambda_gp)
    celoss = CELoss()
    ganloss = GANLoss()
    canloss = CANLoss(celeba.classes)
    pixelwiseloss = PixelwiseLoss(opts.lambda_photo)

    for epoch in range(opts.num_epoch):
        for i, (img, cls, bbox, landm) in enumerate(dataloader):
            img = img.to(device)
            cls = cls.to(device)
            bbox = bbox.to(device)
            landm = landm.to(device)

            # preprocess
            img_ = img.clone()
            masked = to_apply_mask(img_, bbox) 
            expanded_bbox = to_expand_bbox(bbox, landm, *img.shape[2:])
            cropped_masked = to_crop_resize(masked, expanded_bbox).to(img_.device)
            cropped_img = to_crop_resize(img, expanded_bbox).to(img.device)

#            # 이미지 저장
#            save_image(restore, 
#                       f'test.jpg',
#                       nrow=opts.batch_size//4, normalize=True)
#            print('saved')

            if i % opts.n_repeat == 0:
                # # # # #
                # Discriminator
                # # # # #
                fake = G(cropped_masked)
                D_fake_cls, D_fake = D(fake)
                D_real_cls, D_real = D(cropped_img)

                loss_D_wgan = wganloss(cropped_img, fake)
                loss_D_cls_real = celoss(D_real_cls, cls)
                loss_D = loss_D_wgan + loss_D_cls_real
                
                D_optim.zero_grad()
                loss_D.backward()
                D_optim.step()

            # # # # #
            # Generator
            # # # # #
            fake = G(cropped_masked)
            D_fake_cls, D_fake = D(fake)
            loss_G_wgan = wganloss(fake)
            loss_G_can = canloss(D_fake_cls)

            restore = to_restore_size(masked, fake, expanded_bbox)
            loss_photorealistic = pixelwiseloss(img, restore)
            loss_G = loss_G_wgan + loss_G_can + loss_photorealistic
            
            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()

            # 진행상황 출력
            softmax_fake_cls = softmax_(D_fake_cls)
            softmax_real_cls = softmax_(D_real_cls)
            print(f"[Epoch {epoch}/{opts.num_epoch}     Batch {i*opts.batch_size}/{len(celeba)}]")
            print(f"D loss: {loss_D:.6f}")
            print(f"G loss: {loss_G:.6f}")
            print(f"            |    real    |    fake    ")
            print(f"--------------------------------------")
            print(f"Min prob    | {softmax_real_cls.min(1)[0].mean():.6f}   |   {softmax_fake_cls.min(1)[0].mean():.6f}")
            print(f"Max prob    | {softmax_real_cls.max(1)[0].mean():.6f}   |   {softmax_fake_cls.max(1)[0].mean():.6f}")
            print(f"Precision   | {correct_(cls, softmax_real_cls):.4f}     |   {correct_(cls, softmax_fake_cls):.4f}")
            print("="*55)


            # 이미지 저장
            if i%(20000//opts.batch_size)== 0:
                save_image(torch.cat([img, restore], dim=3), 
                f'images/ep{epoch+1:03d}_iter{i*opts.batch_size:05d}.jpg',
                nrow=opts.batch_size//4, normalize=True)

        torch.save({
            'G' : G.state_dict(),
            'D' : D.state_dict(),
            'opts' : opts
            },
            'pretrained/ppad_v2.pth')

if __name__ == '__main__':
    main()
