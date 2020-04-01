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
from models.resnet import resnet18
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
parser.add_argument('--lr', type=int, default=1e-4)
parser.add_argument('--b1', type=int, default=0.5)
parser.add_argument('--b2', type=int, default=0.999)
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
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

### celeba 데이터셋
celeba = celebA(path='../data/celebA_aligned/',
                transforms=transforms)

### 데이터 로드
dataloader = DataLoader(celeba,
                        batch_size=opts.batch_size,
                        shuffle=True)


def main():
    if opts.net == 'resnet18':
        # 네트워크
        D = resnet18(num_classes=celeba.classes).to(device)
    else:
        raise NotImplemented

    # pretrained 모델 불러오기
    if opts.reuse:
        assert os.path.isfile('pretrained/ppad_v2.pth')
        checkpoint = torch.load(ckptname)
        D.load_state_dict(checkpoint['D'])
        print('[*]Pretrained model loaded')

    # optimizer 정의
    D_optim = optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))

    # loss 정의
    mseloss = nn.MSELoss()
    celoss = nn.CrossEntropyLoss()

#    # scheduler
#    scheduler = optim.lr_scheduler.StepLR(D_optim, step_size=opts.lr_step ,gamma=0.1)

    for epoch in range(opts.num_epoch):
        for i, (img, cls, bbox, landm) in enumerate(dataloader):
            img = img.to(device)
            cls = cls.to(device)
            bbox = bbox.to(device)
            landm = landm.to(device)

            expanded_bbox = to_expand_bbox(bbox, landm, *img.shape[2:])
            cropped_img = to_crop_resize(img, expanded_bbox).to(device)

            D_real_cls, D_real = D(cropped_img)
            D_cls_loss = celoss(D_real_cls, cls)
            D_real_loss = mseloss(D_real, torch.ones_like(D_real))
            loss_D = D_cls_loss + D_real_loss 

            D_optim.zero_grad()
            loss_D.backward()
            D_optim.step()

            # 진행상황 출력
            softmax_real_cls = softmax_(D_real_cls)
            minprob = softmax_real_cls.min(1)[0].mean()
            maxprob = softmax_real_cls.max(1)[0].mean()
            acc = accuracy_(cls, softmax_real_cls)
            print(f"[Epoch {epoch}/{opts.num_epoch}] [Batch {i*opts.batch_size}/{len(celeba)}]")
            print(f"[D loss {loss_D:.6f}] [D_real_cls {D_cls_loss:.6f}] [D_real {D_real_loss:.6f}]")
            print(f"Min prob    | {minprob:.6f}")
            print(f"Max prob    | {maxprob:.6f}")
            print(f"Precision   | {acc:.4f}")
            print(f"True/False  | {torch.sum(D_real)/len(D_real):.2f}")
            print("="*55)

            # 로깅
            if i % (len(dataloader) / opts.logstep_per_epoch) == 0 and opts.logging:
                logger.write('epoch',epoch, 'iter',i, 'loss_d',loss_D, 'd_cls_loss', D_cls_loss, 
                             'd_real_loss', D_real_loss, 'acc', acc, 'minprob', minprob, 'maxprob', maxprob)


        torch.save({
            'D' : D.state_dict(),
            },
            f'pretrained/trained_D.pth')

    logger.close()

if __name__ == '__main__':
    main()
