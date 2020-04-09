from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
import argparse
import time
import os

from models.v1 import weights_init
from models.resnet_ import resnet_face18
from models.focal_loss import FocalLoss
from models.metrics import ArcMarginProduct
from data.celeba_aligned import celebA
from logger import Logger
from utils import *

now = time.localtime()

# 파라매터
parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='resnet18')
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--lr_decay', type=float, default=0.95)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--reuse', dest='reuse', action='store_true')
parser.set_defaults(reuse=False)
parser.add_argument('--logging', dest='logging', action='store_true')
parser.set_defaults(logging=False)
parser.add_argument('--logstep_per_epoch', type=int, default=5)
parser.add_argument('--filename', type=str, default='{:4d}{:02d}{:02d},{:02d}{:02d}'.format(
                    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min))
opts = parser.parse_args()
print(opts)

# 저장공간생성
os.makedirs('pretrained', exist_ok=True)

# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 로그 생성
if opts.logging:
    logger = Logger(opts.filename+'.txt')
    logger.write(opts)

# 이미지 불러오기
### 이미지 전처리
transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])

### celeba 데이터셋
celeba = celebA(path='../../myProject/GANs-pytorch/data/celeba/',
                transforms=transforms,
                )

### 데이터 로드
dataloader = DataLoader(celeba,
                        batch_size=opts.batch_size,
                        shuffle=True)


def main():
    # 네트워크
    D = resnet_face18(use_se=False).to(device)
    arcmargin = ArcMarginProduct(512, celeba.classes, s=30, m=0.5, easy_margin=False).to(device)
    
    # pretrained 모델 불러오기
    if opts.reuse:
        assert os.path.isfile('pretrained/trained_D.pth')
        checkpoint = torch.load('pretrained/trained_D.pth')
        D.load_state_dict(checkpoint['D'])
        arcmargin.load_state_dict(checkpoint['arcmargin'])
        print('[*]Pretrained model loaded')

    # optimizer 정의
    D_optim = optim.SGD([{'params':D.parameters()}, {'params':arcmargin.parameters()}],
                        lr=opts.lr, weight_decay=opts.weight_decay)

    # loss 정의
    criterion = FocalLoss(gamma=2)
    mseloss = nn.MSELoss()

    # scheduler
    scheduler = optim.lr_scheduler.StepLR(D_optim, step_size=opts.lr_step ,gamma=0.1)

    for epoch in range(opts.num_epoch):
        for i, (img, cls, bbox, landm) in enumerate(dataloader):
            img = img.to(device)
            cls = cls.to(device)
            bbox = bbox.to(device)
            landm = landm.to(device)

            expanded_bbox = to_expand_bbox(bbox, landm, *img.shape[2:])
            cropped_img = to_crop_resize(img, expanded_bbox).to(device)

            feature = D(cropped_img)
            output = arcmargin(feature, cls)
            loss_D = criterion(output, cls)

            D_optim.zero_grad()
            loss_D.backward()
            D_optim.step()

            # 진행상황 출력
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            cls = cls.data.cpu().numpy()
            acc = np.mean((output==cls).astype(int)).astype(float)
            print(f"[Epoch {epoch}/{opts.num_epoch}] [Batch {i*opts.batch_size}/{len(celeba)}] [D loss {loss_D:.6f}] lr {scheduler.get_lr()}")
            print(f"Precision   | {acc:.4f}")
            print("="*65)

            # 로깅
            if i % (len(dataloader) // opts.logstep_per_epoch) == 0 and opts.logging:
                logger.write('epoch',epoch, 'iter',i, 'loss_d',loss_D, 'acc', acc)

        torch.save({
            'D' : D.state_dict(),
            'arcmargin' : arcmargin.state_dict()
            },
            'pretrained/trained_D_arc.pth')

    scheduler.step()

if __name__ == '__main__':
    main()
