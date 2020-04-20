"""Deep Q Network(based on the atari framework)


A CNN with 3 convolution networks follwed by 2 linear layers.Here we're
not using any pooling layers since we don't want the image to be transition
variant. For more refer
https://www.intel.com/content/www/us/en/artificial-intelligence/posts/demystifying-deep-reinforcement-learning.html
"""
from __future__ import print_function

import torch,torchvision
import cv2
import sys
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


output_size = 2

class DQN(nn.Module):

    def __init__(self, h, w, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)


        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc2(self.fc1(x.view(x.size(0), -1)))
