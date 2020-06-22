#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class Highway(nn.Module):
    #pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.proj_projection = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.gate_projection = nn.Linear(word_embed_size, word_embed_size, bias=True)
        #self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x_convout):
        self.x_proj = F.relu(self.proj_projection(x_convout))
        #print("x_proj shape: ", self.x_proj.shape)

        self.x_gate = torch.sigmoid(self.gate_projection(x_convout))
        #print("x_gate shape: ", self.x_gate.shape)

        self.x_highway = self.x_gate * self.x_proj + (1-self.x_gate) * x_convout
        #print("x_highway shape: ", self.x_highway.shape)

        #self.x_word_emb = self.dropout(self.x_highway)
        #print("x_word_emb shape: ", self.x_word_emb.shape)

        return self.x_highway
    ### END YOUR CODE

