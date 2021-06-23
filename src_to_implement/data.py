import skimage
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode,transform=None):
        self.data = data
        self.mode = mode
        self._transform = transform
        if self.mode == 'train':
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),tv.transforms.Normalize(train_mean,train_std)])
        elif self.mode == 'val':
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),tv.transforms.Normalize(train_mean,train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_gray = skimage.io.imread(self.data.iloc[index, 0])
        image_rgb = skimage.color.gray2rgb(image_gray)
        y_label = np.array(self.data.iloc[index, 1:3],dtype=float)
        if self._transform:
            image_rgb = self._transform(image_rgb)
            y_label = torch.from_numpy(y_label)
        return image_rgb,y_label





