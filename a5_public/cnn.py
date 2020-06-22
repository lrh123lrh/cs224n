#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    #pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, input_size, filter_size, kernel_size=5):
        super(CNN, self).__init__()
        #kernel_size = 5
        padding = 1
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=filter_size, kernel_size=kernel_size, padding=padding)

    def forward(self, x_reshaped):
        hidden1 = self.cnn(x_reshaped)
        hidden2 = torch.max(F.relu(hidden1), dim=2)[0]# torch.max returns a tuple (max_values, indices)
        return hidden2
    ### END YOUR CODE

