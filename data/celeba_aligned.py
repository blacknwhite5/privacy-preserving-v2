from torch.utils.data import Dataset 
import torch

from PIL import Image
import os.path as osp
import numpy as np

class celebA(Dataset):
    def __init__(self, path, mode='total', transforms=None, grayscale=False):
        self.path = path
        self.transforms = transforms
        self.mode = mode
        self.grayscale = grayscale

        self.imgfolder = 'img_align_celeba'
        self.cache = 'cache.txt'

        # make or read cache 
        if osp.isfile(osp.join(self.path, self.cache)):
            self.data = self.read_cache()
        else:
            self.data = self.make_cache()

        # split train/valid/test
        if self.mode == 'total':
            pass
        else:
            self.data = self.split(self.mode)

        # get imgnames & number of classes
        self.imgnames = list(self.data.keys())
        labels = []
        for imgname in self.imgnames:
            cls = self.data[imgname][0]
            labels.append(cls)

        self.classes = len(set(labels))
        print('[*] celeba data loaded')


    def __getitem__(self, idx):
        imgname = self.imgnames[idx]
        cls, box, landm = self.data[imgname]
        img = Image.open(osp.join(self.path, self.imgfolder, imgname)).convert('RGB')
        if self.grayscale:
            img = img.convert('L')

        if torch.rand(1) > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            # flip landm
            x1, x2, x3, x4, x5 = landm[0::2]
            landm[0] = w - x2 
            landm[2] = w - x1
            landm[4] = w - x3
            landm[6] = w - x5
            landm[8] = w - x4
            # flip box
            x_max = w - box[0]
            x_min = w - box[2]
            box[0] = max(x_min, 0)
            box[2] = min(x_max, w)

        img = self.transforms(img)
        cls = torch.tensor(cls)
        box = torch.tensor(box)
        landm = torch.tensor(landm)
        return img, cls, box, landm

    def __len__(self):
        return len(self.imgnames)

    def make_cache(self):
        print("[*] Making cache...")
        # read bbox, landm info
        f = open(osp.join(self.path, 'celeba_kpnts.txt'), 'r')
        info = {}
        for s in f.read().split('\n')[1:-1]:
            data = s.split(',')
            imgname = data[0]
            box = list(map(int, data[1:5]))
            landm = list(map(int, data[5:]))
            info[imgname] = [box, landm]
        f.close()
        
        # read label info
        labels = {}
        f = open(osp.join(self.path, 'identity_CelebA.txt'), 'r')
        for s in f:
            filename, identity = s.split()
            labels[filename] = int(identity)-1
        f.close()

        data = {}
        f = open(osp.join(self.path, self.cache), 'w')
        for imgname in info.keys():
            box, landm = info[imgname]
            cls = labels[imgname]
            data[imgname] = [cls, box, landm]
            f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(imgname, cls, *box, *landm))
        f.close()

        del info
        del labels

        print("[*] Making done!!")
        return data

    def read_cache(self):
        print("[*] Reading cache...")
        f = open(osp.join(self.path, self.cache), 'r')
        data = {}
        for dat in f.read().split('\n')[:-1]:
            dat = dat.split(',')
            imgname = dat[0]
            cls = int(dat[1])
            box = list(map(int, dat[2:6]))
            landm = list(map(int, dat[6:]))
            data[imgname] = [cls, box, landm]
        print("[*] Reading done!")
        return data
            

    def split(self, mode):
        """split train/valid/test"""
        if mode == 'train':
            mode = 0
        elif mode == 'valid':
            mode = 1
        elif mode == 'test':
            mode =2
        else:
            raise Exception(f"need to 'train', 'valid', 'test' for dividing dataset, but got {mode}")

        phase = {}
        f = open(osp.join(self.path, 'list_eval_partition.txt'))
        for s in f.read().split('\n')[:-1]:
            imgname, party = s.split()
            phase[imgname] = int(party)
        f.close()

        data = {}
        for imgname in self.data.keys():
            cls, bbox, landm = self.data[imgname]
            party = phase[imgname]
            if party == mode:
                data[imgname] = [cls, bbox, landm]
        return data


if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])])
    
    celeba = celebA('/home/moohyun/Desktop/myProject/GANs-pytorch/data/celeba/', transforms=transform)
    dataloader = DataLoader(celeba, batch_size=16, shuffle=True)


    for i, (img, cls, bbox, landm) in enumerate(dataloader):
        print(img.shape, cls.shape, bbox.shape, landm.shape)
        pass 


#        # check pillow image
#        draw = ImageDraw.Draw(img)
#        draw.rectangle(bbox)
#        for i in range(5):
#            draw.ellipse((landm[i*2]-1, landm[i*2+1]-1, landm[i*2]+1, landm[i*2+1]+1), fill=(255,0,0,0))
#        img.save('flip.jpg')
