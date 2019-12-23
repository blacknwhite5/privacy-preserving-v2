import os
import glob
from PIL import Image
from torch.utils.data import Dataset 

class celebA(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode

        # check images
        self.images = glob.glob(os.path.join(self.data_path, 'img_align_celeba/**/*.jpg'), recursive=True)
        
        # check labels
        self.labels = {}
        f = open(os.path.join(self.data_path, 'identity_CelebA.txt'), 'r')
        for s in f:
            filename, identity = s.split()
            self.labels[filename] = int(identity)
        f.close()

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = self.transform(img)
        label = self.labels[self.images[idx].split('/')[-1]]
        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms 
    transform = transforms.Compose([transforms.ToTensor()])
    celeba = celebA('../../../myProject/GANs-pytorch/data/celeba/', transform=transform)
    dataloader = DataLoader(dataset=celeba, batch_size=1, shuffle=False)

    for img, label in dataloader:
        print(img.shape, label)
