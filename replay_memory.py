from collections import namedtuple
import random
import torch
import numpy

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_memory(self):
        return self.memory

    def set_memory(self, memory):
        self.memory = []
        self.position = 0
        for i in range(len(memory)):
            if memory[i] is not None:
                self.push(*memory[i])
