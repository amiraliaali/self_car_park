import torch
import random
from collections import deque


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)

        # [[s, a, r, s'], [s, a, r, s'], [s, a, r, s']] -> [[s, s, s], [a, a, a], [r, r, r], [s', s', s']]
        batch = zip(*batch)

        # Stack instead of concatenate, keeping dimensions consistent for states
        return [torch.stack(items) for items in batch]


    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10 # the number of the transitions in the memory should be at least 10 times the batch size

    def __len__(self):
        return len(self.memory)