from os import listdir
from os.path import join

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class dataloader(data.Dataset):
    def __init__(self, txt_path, data_path, preproc=None, mode='training'):
        self.preproc = preproc
        self.imgs_path = []
        self.labels = []
        self.weights = []
        self.names = []
        self.mode = mode
        if self.mode == 'training' or self.mode == 'validate':
            df = pd.read_csv(txt_path)
            tmp = []
            for i in range(5):
                tmp = df.level[df.level == i].count()
                self.weights.append(float(len(df.image)/(5*tmp)))
            img_list = list(df['image'])
            self.weights = torch.tensor(self.weights).float()
        else:
            img_list = listdir(data_path)
        for i in img_list:
            image_name = i if '.jpeg' in i else i+'.jpeg'
            self.names.append(i[:-5])
            self.imgs_path.append(join(data_path, image_name))
            if self.mode == 'training' or self.mode == 'validate':
                tmp_label = int(df.level[df['image'] == i])
                self.labels.append(tmp_label)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        if self.mode == 'training' or self.mode == 'validate':
            label = self.labels[index]
            img, target = self.preproc(img, label, self.mode)
        else:
            target = self.names[index]
            img, target = self.preproc(img, target, self.mode)
        return torch.from_numpy(img.transpose(2, 0, 1)), target
