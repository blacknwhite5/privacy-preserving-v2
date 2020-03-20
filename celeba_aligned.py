import os
import glob
from PIL import Image
from torch.utils.data import Dataset 

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


    def apply_mask_bbox(self, img, bbox):
        """mask bbox of image"""
        x1, y1, x2, y2 = bbox
        masked_img = img.clone()
        masked_part = img[:,y1:y2,x1:x2]
        masked_img[:,y1:y2,x1:x2] = 1 
        return masked_img, masked_part

    def __getitem__(self, idx):
        label = self.labels[self.imgnames[idx].split('/')[-1]]
        box = self.bboxes[idx]
        landm = self.landmarks[idx]

        img = Image.open(os.path.join(self.path, self.imgfolder, self.imgnames[idx]))
        img = self.transforms(img)
        masked_img, masked_part = self.apply_mask_bbox(img, box)
        return img, masked_img, masked_part, label, landm

    def __len__(self):
        return len(self.imgnames)
