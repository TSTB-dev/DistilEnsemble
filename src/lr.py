
import numpy as np

class CyclicCosineLRScheduler:
    def __init__(self, optimizer, max_lr, min_lr, cycle_length, cycle_length_decay=1):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length
        self.cycle_length_decay = cycle_length_decay
        self.iteration = 0
        self.cycle = 0
        self.cycle_position = 0
        self.update_lr()
    
    def update_lr(self):
        cycle_pos = self.cycle_position / self.cycle_length
        lr = self.max_lr * 0.5 * (1 + np.cos(np.pi * cycle_pos))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self, optim_step=False):
        self.update_lr()
        if optim_step:
            self.optimizer.step()
        self.iteration += 1
        self.cycle_position += 1
        if self.cycle_position >= self.cycle_length:
            self.cycle_position = 0
            self.cycle_length *= self.cycle_length_decay
            self.cycle += 1
        return self.cycle_position == 0
    

    