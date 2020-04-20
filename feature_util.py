import os
import numpy as np
import cv2
import torch
from os import listdir

from torchvision.transforms import ToTensor


DIM = 84

def generate_state(image_arrays):
    state = torch.empty(4,84,84)
    for img in image_arrays:
        # img = cv2.imread(img) #Add if filepaths are passed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(DIM,DIM))
        state[image_arrays.index(fp)] = ToTensor()(img)
