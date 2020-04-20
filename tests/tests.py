import numpy as np
import torch
import sys
import os
sys.path.append(os.getcwd())

# import dqn
# from .. import dqn
from dqn import DQN


def dqn_forward_testing():
    input = torch.rand((1,3,84,84))
    h,w = 84,84
    output_size = 2
    net = DQN(h,w,2)
    res = net(input)
    if res.shape[0] == 1 and  res.shape[1] == 2:
        print("dqn_forward_testing for output shape is Correct")
    else:
        print("dqn_forward_testing for output shape is InCorrect")

if __name__=="__main__":
    dqn_forward_testing()
