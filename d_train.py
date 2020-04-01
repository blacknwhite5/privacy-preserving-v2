from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
import os

from models.v1 import weights_init
from models.resnet_ import resnet18, resnet_face18
from models.focal_loss import FocalLoss
from models.metrics import ArcMarginProduct
from data.celeba_aligned import celebA
from utils import *


# 파라매터
reuse = False
num_epoch = 100
batch_size = 16 
lr = 0.05 
lr_step = 10
lr_decay = 0.95
weight_decay = 5e-4

# 저장공간생성
os.makedirs('pretrained', exist_ok=True)

# GPU 사용여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 불러오기
### 이미지 전처리
transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])

### celeba 데이터셋
celeba = celebA(path='../../myProject/GANs-pytorch/data/celeba/',
                transforms=transforms,
                grayscale=True)

### 데이터 로드
dataloader = DataLoader(celeba,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=2)


def main():
    # 네트워크
    D = resnet_face18(in_channel=1, use_se=False).to(device)
    arcmargin = ArcMarginProduct(512, celeba.classes, s=30, m=0.5, easy_margin=False).to(device)
    
    # pretrained 모델 불러오기
    if reuse:
        assert os.path.isfile('pretrained/trained_D.pth')
        checkpoint = torch.load('pretrained/trained_D.pth')
        D.load_state_dict(checkpoint['D'])
        arcmargin.load_state_dict(checkpoint['arcmargin'])
        print('[*]Pretrained model loaded')

    # optimizer 정의
    D_optim = optim.SGD([{'params':D.parameters()}, {'params':arcmargin.parameters()}],
                        lr=lr, weight_decay=weight_decay)

    # loss 정의
    criterion = FocalLoss(gamma=2)
    mseloss = nn.MSELoss()

    # scheduler
    scheduler = optim.lr_scheduler.StepLR(D_optim, step_size=lr_step ,gamma=0.1)

    for epoch in range(num_epoch):
        scheduler.step()

        for i, (img, cls, bbox, landm) in enumerate(dataloader):
            img = img.to(device)
            cls = cls.to(device)
            bbox = bbox.to(device)
            landm = landm.to(device)

            expanded_bbox = to_expand_bbox(bbox, landm, *img.shape[2:])
            cropped_img = to_crop_resize(img, expanded_bbox).to(device)

            feature, D_real = D(cropped_img)
            output = arcmargin(feature, cls)
            arcloss = criterion(output, cls)
            realloss = mseloss(D_real, torch.ones_like(D_real))
            loss_D = arcloss + realloss

            D_optim.zero_grad()
            loss_D.backward()
            D_optim.step()

            # 진행상황 출력
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            cls = cls.data.cpu().numpy()
            acc = np.mean((output==cls).astype(int)).astype(float)

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Acc: %f] [sample %d %d %d | %d %d %d]" % \
                  (epoch, num_epoch, i*batch_size, len(celeba), loss_D.item(), acc, 
                   output[0], output[1], output[1], cls[0], cls[1], cls[2]))

        torch.save({
            'D' : D.state_dict(),
            'arcmargin' : arcmargin.state_dict()
            },
            'pretrained/trained_D_arc.pth')

if __name__ == '__main__':
    main()
