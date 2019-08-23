import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class DataFolder(Dataset):
    """
        DataFolder: Creates batches of tensors for training and validation
        Params:
            * root      :   path of the root directory where the data is stored in 
                            folders 'train' and 'val'
            * phase     :   switch between training and validation phase
            * transform :   Transformations done on the acquired images
            * blurStat  :   Flag to enable Gaussian Blur
            * ksize     :   kernel size for Gaussian Blur
    """
    def __init__(self, root, phase, transform=None, blurStat=False, ksize=15):
        self.root_dir = os.path.join(root, phase)
        self.listIdx = [os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir)]
        self.transform = transform
        self.blurStat = blurStat
        self.ksize = ksize

    def __len__(self):
        """
            Args    : dataset object
            Returns : length of dataset
        """
        return (len(self.listIdx))

    def __getitem__(self, idx):
        """
            Args    : dataset object, index
            Returns : 'idx'th element in the batch
        """
        img = cv2.imread(self.listIdx[idx])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if self.blurStat == True:
            img = cv2.GaussianBlur(img, (self.ksize, self.ksize), 1) 
        if self.transform:
            img = self.transform(img)
        return img

class Options():

    def __init__(self, data, nz, nf, n_extra_layers):
        self.isize = data.size()[-1]
        self.nc    = data.size()[1]
        self.nz    = nz
        self.ngf = nf
        self.ndf = nf
        self.n_extra_layers = n_extra_layers
