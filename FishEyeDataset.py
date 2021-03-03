# coding=utf-8

import os, torch
import cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader


class FishEyeDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(FishEyeDataset, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.file_root = root + '/鱼眼视频_训练集/'
        fisheye = root + '/鱼眼视频_训练集/fisheye/'
        nofisheye = root + '/鱼眼视频_训练集/no_fisheye/'


        self.filenames = []
        self.labels = []

        for file in os.listdir(fisheye):
            if file[-4:] == '.jpg':
                self.filenames.append('/fisheye/' + file)
                self.labels.append(torch.tensor(1))

        for file in os.listdir(nofisheye):
            if file[-4:] == '.jpg':
                self.filenames.append('/no_fisheye/' + file)
                self.labels.append(torch.tensor(0))

    def __getitem__(self, index):
        img_name = self.file_root + self.filenames[index]
        label = self.labels[index]
        img = cv2.imread(img_name)
        img = cv2.resize(img, (336, 192))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float()
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filenames)

