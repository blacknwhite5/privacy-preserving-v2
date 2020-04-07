from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os


class CASIA(Dataset):
    def __init__(self, path, mode='train', transforms=None, size=(128,128)):
        self.path = path
        self.transforms = transforms
        self.mode = mode
        self.size = size
        self.imgfolder = 'CASIA-WebFace'
        self.cache = 'cache.txt'

        if os.path.isfile(os.path.join(self.path, self.cache)):
            self.data = self.read_cache()
        else:
            self.data = self.make_cache()
        self.filenames = list(self.data.keys())

        self.classes = []
        for imgname in self.data.keys():
            self.classes.append(self.data[imgname][0])

        self.classes = len(set(self.classes))
        print('[*] CASIA dataset loaded')


    def __getitem__(self, idx):
        imgname = self.filenames[idx]
        cls, box, landm = self.data[imgname]
        img = Image.open(os.path.join(self.path, self.imgfolder, imgname)).convert('RGB')
        cls = np.array(cls)
        box = np.array(box)
        landm = np.array(landm)

        # resize (250 -> 128)
        # img, box, landm = self.resize(img, box, landm)

        # stochastic flip 
        if torch.rand(1) > 0.5:
            box = box.copy()
            landm = landm.copy()

            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            # flip landm
            x1, x2, x3, x4, x5 = landm[0::2]
            landm[0] = w - x2
            landm[2] = w - x1
            landm[4] = w - x3
            landm[6] = w - x5
            landm[8] = w - x4
            # flip bbox
            x_max = w - box[0]
            x_min = w - box[2]
            box[0] = max(x_min, 0)
            box[2] = min(x_max, w)
        
        img = self.transforms(img)
#        cls = torch.tensor(cls)
#        box = torch.tensor(box)
#        landm = torch.tensor(landm)
        return img, cls, box, landm
        

    def __len__(self):
        return len(self.data)


    def make_cache(self):
        print("[*] Making cache...")
        f = open(os.path.join(self.path, 'casia_landmark.txt'), 'r')
        cls_n_landmark = {}
        for dat in f.read().split('\n')[:-1]:
            dat = dat.split('\t')
            imgname = dat[0]
            cls = int(dat[1])
            landm = list(map(int, dat[2:]))
            cls_n_landmark[imgname] = [cls, landm]
        f.close()

        f = open(os.path.join(self.path, 'MM16Annotation-CASIAWebFace.txt'), 'r')
        bbox = {}
        for dat in f.read().split('\n')[:-1]:
            dat = dat.split()
            imgname = dat[0][1:]
            box = list(map(int, dat[1:]))
            box = self.xywh2xyxy(box)
            bbox[imgname] = box
        f.close()
            
        s1 = set(cls_n_landmark.keys())
        s2 = set(bbox.keys())
        intersection = s1 & s2
        data = {}
        for imgname in intersection:
            box = bbox[imgname]
            cls, landm = cls_n_landmark[imgname]
            data[imgname] = [cls, box, landm]
        del cls_n_landmark
        del bbox

        f = open(os.path.join(self.path, 'cache.txt'), 'w')
        for key, value in data.items():
            cls, box, landm = value 
            f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(key, cls, *box, *landm))
        f.close()
        print("[*] Making done!!")
        return data

    def read_cache(self):
        print("[*] Reading cache...")
        data = {}
        with open(os.path.join(self.path, self.cache)) as f:
            for info in f.read().split('\n')[:-1]:
                info = info.split(',')
                imgname = info[0]
                cls = int(info[1])
                box = list(map(int, info[2:6]))
                landm = list(map(int, info[6:]))
                data[imgname] = [cls, box, landm]
        print("[*] Reading done!")
        return data

    def xywh2xyxy(self, box):
        x, y, w, h = box
        return [x, y, x+w, y+h]

    def resize(self, img, box, landm):
        """resize (250 -> 128)"""
        box = np.array(box)
        landm = np.array(landm)

        scale = self.size[0] / img.size[0]
        img = img.resize(self.size, Image.LANCZOS)
        box = np.round(box*scale).astype(box.dtype)
        landm = np.round(landm*scale).astype(landm.dtype)
        return img, box, landm
        



if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])])
    
    casia = CASIA('/home/moohyun/Desktop/egovid/PPAD/data/CASIA/', transforms=transform)
    dataloader = DataLoader(casia, batch_size=16, shuffle=True)
       
    for i, (img, cls, bbox, landm) in enumerate(dataloader):
        print(i, img.shape, cls.shape, bbox.shape, landm.shape)
        pass 

#        # check pillow image
#        draw = ImageDraw.Draw(img)
#        draw.rectangle(list(box))
#        for i in range(5):
#            draw.ellipse((landm[i*2]-1, landm[i*2+1]-1, landm[i*2]+1, landm[i*2+1]+1), fill=(255,0,0,0))
#        img.save('test.jpg')
