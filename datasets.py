import os
import glob
from PIL import Image
from torch.utils.data import Dataset 

class celebA(Dataset):
    def __init__(self, path, mode='train', transforms=None):
        self.path = path
        self.transforms = transforms
        self.mode = mode

        # check images
        self.images = glob.glob(os.path.join(self.path, 'img_align_celeba/**/*.jpg'), recursive=True)
        
        # check labels
        self.labels = {}
        f = open(os.path.join(self.path, 'identity_CelebA.txt'), 'r')
        for s in f:
            filename, identity = s.split()
            self.labels[filename] = int(identity)-1
        f.close()
        
        # num of classes
        self.classes = len(set(self.labels.values()))

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = self.transforms(img)
        label = self.labels[self.images[idx].split('/')[-1]]
        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    data = celebA('../../../myProject/GANs-pytorch/data/celeba/')

