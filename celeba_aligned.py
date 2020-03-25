from torch.utils.data import Dataset 
from PIL import Image
import numpy as np
import torch
import os

class celebA(Dataset):
    def __init__(self, path, mode='train', transforms=None):
        self.path = path
        self.transforms = transforms

        self.mode = mode
        self.imgfolder = 'img_align_celeba'

        # check bbox, landm
        f = open(os.path.join(self.path, 'celeba_kpnts.txt'), 'r')
        self.imgnames, self.bboxes, self.landmarks = [], [], []
        for s in f.read().split('\n')[1:-1]:
            data = s.split(',')
            self.imgnames.append(data[0])
            self.bboxes.append(list(map(int, data[1:5])))
            self.landmarks.append(list(map(int, data[5:])))
        f.close()
        
        # check labels
        self.labels = {}
        f = open(os.path.join(self.path, 'identity_CelebA.txt'), 'r')
        for s in f:
            filename, identity = s.split()
            self.labels[filename] = int(identity)-1
        f.close()
        
        # num of classes
        self.classes = len(set(self.labels.values()))
        print('[*] celeba data loaded')


    def __getitem__(self, idx):
        """original, expaned bbox with replace 128 in face area, cls, landm"""
        label = self.labels[self.imgnames[idx].split('/')[-1]]
        bbox = self.bboxes[idx]
        landm = self.landmarks[idx]

        img = Image.open(os.path.join(self.path, self.imgfolder, self.imgnames[idx]))
        img = self.transforms(img)
        bbox = torch.tensor(bbox)
        landm = torch.tensor(landm)
        return img, label, bbox, landm

    def __len__(self):
        return len(self.imgnames)



if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])])
    
    celeba = celebA('../data/celebA_aligned/', transforms=transform)
    dataloader = DataLoader(celeba, batch_size=16, shuffle=True)

    for i, (img, cls, bbox, landm) in enumerate(dataloader):
        print(img.shape, cls, bbox, landm)
        pass 
