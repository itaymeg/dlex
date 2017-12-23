#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torchfile
import os
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

dtype = torch.LongTensor
#load data
data_dir = './data/'
train = torchfile.load(os.path.join(data_dir, 'train.t7')) #load train data
vocab = torchfile.load(os.path.join(data_dir, 'vocab.t7')) #load vocabulary mapping char -> index
vocab_rev = {a:b for b,a in vocab.items()} # index -> char
train = Variable(torch.from_numpy(train)).type(dtype)
labels = train[1:]


class CharRNN(nn.Module):
    def __init__(self, use_cuda = False, vocab_size = 35, embed_size = 1, hidden_size = 64, n_layers = 128):
        super(CharRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax()
        self.use_cuda = use_cuda
    def forward(self, x):
        print(x.size())
        x = x.view(128,1)
        #mbsize x 
        x = self.embed(x)
        output, (hidden, cell) = self.lstm(x)
        print('Output ', output.size(), ' Hidden ', hidden.size())
        x = self.fc(hidden)
        print('FC ' , x.size())
        x = self.softmax(x)
        print('Softmax ' , x.size())
        print(x)
        x= x.view(128,35)
        values, indices = torch.max(x,dim= 1) # by rows
        print('Indices ' , indices.size())
        print(indices)
        return indices

net = CharRNN()
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
train_mod = train[:len(train) - len(train)%128]
for epoch in tqdm(range(10)):
    for i in range(0,len(train_mod),128):
        optimizer.zero_grad()
        batch = train_mod[i:i+128]
        output = net(batch).type(dtype)
        label = labels[i:i+128].type(dtype)
        output = output.unsqueeze(0)
        label = label.unsqueeze(0)
        print('Output ', output[:5], ' Label ', label[:5])
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

# TODO : Fix Bugs of Loss        
# TODO : Random Crops, Mini- Batches for faster work



        
        
        

#define model

#train model