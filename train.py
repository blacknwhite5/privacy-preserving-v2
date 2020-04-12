from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
import argparse
import time
import os

from models.v1 import resnet_9blocks, PatchGAN, weights_init
from models.resnet import resnet18
from data.celeba_aligned import celebA
from data.casia import CASIA 
from logger import Logger
from utils import *
from loss import *

now = time.localtime()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='celeba')
parser.add_argument('--dnet', type=str, default='resnet18')
parser.add_argument('--gnet', type=str, default='resnet_9blocks')
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--lambda_photo', type=int, default=100)
parser.add_argument('--lambda_gp', type=int, default=10)
parser.add_argument('--n_repeat', type=int, default=1)
parser.add_argument('--reuse', dest='reuse', action='store_true')
parser.set_defaults(reuse=False)
parser.add_argument('--ckpt_d', type=str, default='.')
parser.add_argument('--ckpt_g', type=str, default='.')
parser.add_argument('--logging', dest='logging', action='store_true')
parser.set_defaults(logging=False)
parser.add_argument('--logstep_per_epoch', type=int, default=100)
parser.add_argument('--filename', type=str, default='{:4d}{:02d}{:02d},{:02d}{:02d}'.format(
                    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min))
opts = parser.parse_args()
print(opts)

# seed 고정
torch.manual_seed(0)
cudnn.benchmark = True

# 저장공간생성
os.makedirs(f'{opts.filename}/images', exist_ok=True)
os.makedirs(f'{opts.filename}/pretrained', exist_ok=True)


# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 로그 생성
if opts.logging:
    logger = Logger(os.path.join(opts.filename, 'log.txt'))
    logger.write(opts)

# 이미지 불러오기
### 이미지 전처리
transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

### celeba 데이터셋
if opts.dataset == 'celeba':
    face_data = celebA(path='../../myProject/GANs-pytorch/data/celeba/', transforms=transforms)
elif opts.dataset == 'casia':
    face_data = CASIA(path='../PPAD/data/CASIA/', transforms=transforms)

### 데이터 로드
dataloader = DataLoader(face_data,
                        batch_size=opts.batch_size,
                        shuffle=True)

def main():
    # 네트워크
    D = resnet18(num_classes=face_data.classes).to(device)
    G = resnet_9blocks().to(device)

    # weights 초기화
    D.apply(weights_init)
    G.apply(weights_init)

    # pretrained 모델 불러오기
    if os.path.isfile(opts.ckpt_d):
        load(D, opts.ckpt_d, 'D')
        print('[*]Discriminator model loaded')
    if os.path.isfile(opts.ckpt_g):
        load(G, opts.ckpt_g, 'G')
        print('[*]Generator model loaded')

    # loss 정의
    wganloss = WGAN_GPLoss(D, opts.lambda_gp)
    canloss = CANLoss(face_data.classes)
    pixelwiseloss = PixelwiseLoss(opts.lambda_photo)
    celoss = nn.CrossEntropyLoss()

    # optimizer 정의
    D_optim = optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))
#    D_optim = optim.Adam([
#                          {'params': D.classification.parameters(), 'lr': opts.lr*10}],
#                            lr=opts.lr, betas=(opts.b1, opts.b2))
    G_optim = optim.Adam(G.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))

    # scheduler
#    scheduler_D = optim.lr_scheduler.ExponentialLR(D_optim, gamma=0.99)
#    scheduler_G = optim.lr_scheduler.ExponentialLR(G_optim, gamma=0.99)
    scheduler_D = optim.lr_scheduler.StepLR(D_optim, step_size=6, gamma=0.1)
    scheduler_G = optim.lr_scheduler.StepLR(G_optim, step_size=6, gamma=0.1)


    for epoch in range(opts.num_epoch):
        for i, (img, cls, bbox, landm, imgname) in enumerate(dataloader):
            img = img.to(device)
            cls = cls.to(device)
            bbox = bbox.to(device)
            landm = landm.to(device)

            # preprocess
            img_ = img.clone()
            masked = to_apply_mask(img_, bbox) 

            # use expand bbox
            expanded_bbox = to_expand_bbox(bbox, landm, *img.shape[2:])
            cropped_masked, landm = to_crop_resize(masked, expanded_bbox, landm)
            cropped_img, _ = to_crop_resize(img, expanded_bbox, landm)
            cropped_masked = cropped_masked.to(img_.device)
            cropped_img = cropped_img.to(img_.device)
            onehot_landm = to_onehot(landm, *cropped_masked.shape[2:]).to(img_.device)

#            # verify landmark
#            npimg = (cropped_masked[0].cpu().detach().numpy() + 1) * 127.5
#            npimg = npimg.transpose((1,2,0)).astype(np.uint8)
#            npmark = landm.cpu().detach().numpy()
#            npimg = cv2.umat(npimg)
#            cv2.circle(npimg, (landm[0][0], landm[0][1]), 3, (0,0,255), 2)
#            cv2.circle(npimg, (landm[0][2], landm[0][3]), 3, (0,0,255), 2)
#            cv2.circle(npimg, (landm[0][4], landm[0][5]), 3, (0,0,255), 2)
#            cv2.circle(npimg, (landm[0][6], landm[0][7]), 3, (0,0,255), 2)
#            cv2.circle(npimg, (landm[0][8], landm[0][9]), 3, (0,0,255), 2)
#            cv2.imwrite('test.jpg', npimg)
#            print(npimg)

            if i % opts.n_repeat == 0:
                # # # # #
                # Discriminator
                # # # # #
                fake = G(cropped_masked, onehot_landm)
                D_fake_cls, D_fake = D(fake, onehot_landm)
                D_real_cls, D_real = D(cropped_img, onehot_landm)

                loss_D_rof = wganloss(cropped_img, fake, onehot_landm)
                loss_D_cls = canloss(D_real_cls, D_fake_cls, cls)
#                loss_D_cls = canloss(D_fake_cls) + celoss(D_real_cls, cls)
#                loss_D_cls = celoss(D_real_cls, cls)
                loss_D = loss_D_rof + loss_D_cls 
                
                D_optim.zero_grad()
                loss_D.backward()
                D_optim.step()

            # # # # #
            # Generator
            # # # # #
            fake = G(cropped_masked, onehot_landm)
            D_fake_cls, D_fake = D(fake, onehot_landm)
            restore = to_restore_size(masked, fake, expanded_bbox)

            loss_G_rof = wganloss(fake, onehot_landm)
            loss_G_cls = canloss(D_fake_cls)
            loss_photorealistic = pixelwiseloss(img, restore)
            loss_G = loss_G_rof + loss_G_cls + loss_photorealistic
            
            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()

            # 진행상황 출력
            loss_D_cls = loss_D_cls.data
            loss_G_cls = loss_G_cls.data
            softmax_fake_cls = softmax_(D_fake_cls).data
            softmax_real_cls = softmax_(D_real_cls).data
            minprob_fake = softmax_fake_cls.min(1)[0].mean().data
            maxprob_fake = softmax_fake_cls.max(1)[0].mean().data
            minprob_real = softmax_real_cls.min(1)[0].mean().data
            maxprob_real = softmax_real_cls.max(1)[0].mean().data
            acc_fake = accuracy_(cls, softmax_fake_cls).data
            acc_real = accuracy_(cls, softmax_real_cls).data
            dfake = torch.mean(D_fake).data
            dreal = torch.mean(D_real).data
            print(f"[Epoch {epoch}/{opts.num_epoch}     Batch {i*opts.batch_size}/{len(face_data)}]")
            print(f"[D lr: {scheduler_D.get_lr()}] [G lr: {scheduler_G.get_lr()}]")
            print(f"[LOSS]  |   total        |  real||fake  |   cls ")
            print(f"------------------------------------------------------")
            print(f"D loss  |   {loss_D:2.6f}    |   {loss_D_rof:2.6f}    |   {loss_D_cls:2.6f}")
            print(f"G loss  |   {loss_G:2.6f}    |   {loss_G_rof:2.6f}    |   {loss_G_cls:2.6f}")
            print(f"------------------------------------------------------")
            print(f"            |    real    |    fake    ")
            print(f"--------------------------------------")
            print(f"Min prob    | {minprob_real:2.6f}   |   {minprob_fake:2.6f}")
            print(f"Max prob    | {maxprob_real:2.6f}   |   {maxprob_fake:2.6f}")
            print(f"Precision   | {acc_real:.4f}     |   {acc_fake:.4f}")
            print(f"real || fake| {dreal:2.4f}     |   {dfake:2.4f}")
            print("="*55)


            # 로그 저장
            if i % (len(dataloader) // opts.logstep_per_epoch) == 0 and opts.logging:
                logger.write('epoch',epoch, 'iter',i, 'loss_d',loss_D, 'loss_g', loss_G, 
                             'loss_d_cls', loss_D_cls, 'loss_g_cls', loss_G_cls, 
                             'loss_d_rof', loss_G_rof, 'loss_g_rof', loss_D_rof,
                             'acc_fake', acc_fake, 'acc_real', acc_real, 
                             'minprob_real', minprob_real, 'maxprob_real', maxprob_real,
                             'minprob_fake', minprob_fake, 'maxprob_fake', maxprob_fake)

            # 이미지 저장
            if i % (20000/opts.batch_size) == 0:
                save_image(torch.cat([img, restore], dim=3), 
                f'{opts.filename}/images/ep{epoch+1:03d}_iter{i*opts.batch_size:05d}.jpg',
                nrow=opts.batch_size//4, normalize=True)

        torch.save({
            'G' : G.state_dict(),
            'D' : D.state_dict(),
            },
            f'{opts.filename}/pretrained/ppad_v2.pth')

        scheduler_D.step()
        scheduler_G.step()

    if opts.logging:
        logger.close()

if __name__ == '__main__':
    main()
