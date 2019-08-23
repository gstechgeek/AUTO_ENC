import os
import cv2
import time
import math
import copy
import torch 
import numbers
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from lib.networks import MSCDAE, Encoder, Decoder
from lib.dataprocessing import DataFolder, Options

# ++++++++++++++++++++++++++++++++++++++++++++++ DATA PREPROCESSING +++++++++++++++++++++++++++++++++++++++++++++++++++

params = {
        'batch_size':8,
        'shuffle':True,
        'num_workers':8
    }

# Training with DAGM Class4 dataset
ROOT_DIR = "/home/geet/Documents/mei_yin_yang/data"
dataset = { x: DataFolder(ROOT_DIR, x, transform=transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.Resize(256),
                                                transforms.ToTensor()]), blurStat=False) for x in ['train', 'val'] }

dataloader  = { x: torch.utils.data.DataLoader(dataset[x], **params) for x in ['train', 'val'] }

# +++++++++++++++++++++++++++++++++++++++++++++++ PARAMETER AND FUNCTION INITIALIZATION ++++++++++++++++++++++++++++++++

device = torch.device("cuda:0")
noise_factor = 0.0
num_epochs = 20
nz = 100
nf = 64
n_extra_layers = 5

opt = Options(next(iter(dataloader['train'])), nz, nf, n_extra_layers)

mscdae = MSCDAE(opt).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(mscdae.parameters(), lr=3e-4)

# +++++++++++++++++++++++++++++++++++++++++++++++ TRAINING LOOP +++++++++++++++++++++++++++++++++++++++++++++++++++++++

LOSS = []
for epoch in range(num_epochs):
    print("++++++++++++++++++++ EPOCH {} ++++++++++++++++++++".format(epoch + 1))
    for i, data in enumerate(dataloader['train']):
        optimizer.zero_grad()
        crrpt_data = torch.clamp(data + noise_factor * torch.tensor(
                            np.random.normal(loc=0.0, scale=1.0, size=data.shape), 
                            dtype=torch.float32), 0.0, 1.0)
        crrpt_data = crrpt_data.to(device)
        real_data = data.to(device)
        output = mscdae(crrpt_data)
        loss = criterion(output, real_data)
        LOSS.append(loss.item())
        loss.backward()
        optimizer.step()
        print("{0}\t|{1}/{2}\tLOSS : {3}".format(i + 1, epoch + 1, num_epochs, loss.item()))

def output(batch):
    # actual input
    idx = np.random.randint(len(batch))
    input = batch[idx]
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(input[0])
    
    #corrupted input
    plt.subplot(1, 3, 2)
    plt.title("Corrupted Input")
    crrpt_data = torch.clamp(input + noise_factor * torch.tensor(
                                np.random.normal(loc=0.0, scale=1.0, size=input.shape), 
                                dtype=torch.float32), 0.0, 1.0)
    plt.imshow(crrpt_data[0])

    # model output
    output = mscdae(batch.to(device))
    img = output[idx].permute(1, 2, 0).cpu().detach().squeeze(-1)

    plt.subplot(1, 3, 3)
    plt.title("Output")
    plt.imshow(img)
    plt.show()
