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
import matplotlib.pyplot as plt
from torch.utils.serialization import load_lua
import torch.nn.functional as F

dtype = torch.LongTensor
#load data
data_dir = './data/'
state_dir = './state/'
# train = torchfile.load(os.path.join(data_dir, 'train.t7')) #load train data
# vocab = torchfile.load(os.path.join(data_dir, 'vocab.t7')) #load vocabulary mapping char -> index
# vocab_rev = {a:b for b,a in vocab.items()} # index -> char
# train = Variable(torch.from_numpy(train)).type(dtype)
train = load_lua(os.path.join(data_dir, 'train.t7'))
vocab = load_lua(os.path.join(data_dir, 'vocab.t7'))



class CharRNN(nn.Module):
    def __init__(self, use_cuda = False, vocab_size = 35, hidden_size = 64, n_layers = 1, embed_size = 512):
        super(CharRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = F.softmax
        self.use_cuda = use_cuda
        self.n_layers = n_layers
        self.hidden_size = hidden_size
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = self.embed(x)
        x = x.view(1, batch_size, -1)
        output, hidden = self.lstm(x, hidden)
        x = output.view(batch_size, -1)
        x = self.fc(x)
        # print('FC ' , x.size())
        x = self.softmax(x)
        # print('Softmax ' , x.size())
        # values, indices = torch.max(x, dim=1)  # max per sample in batch
        # # print('Indices ', indices.size())
        return x, hidden

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda())
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

def getData(samples, sentence_len, batch_size):
    x = torch.LongTensor(batch_size, sentence_len)
    y = torch.LongTensor(batch_size, sentence_len)

    for i in range(batch_size):
        randIdx = np.random.randint(0, samples.size(0) - sentence_len -1)
        sentence = samples[randIdx: randIdx + sentence_len + 1]
        x[i, :] = sentence[:-1]
        y[i, :] = sentence[1:]
    return Variable(x), Variable(y)
def save_state(net, epoch):
    file = os.path.join(state_dir, 'state_{0}.pt'.format(epoch))
    net.save_state_dict(file)

hidden_size = 512
sentence_len = 128
batch_size = 128
embed_size = hidden_size
use_cuda = False
vocab_size = len(vocab)
layers = 1
epochs = 5000

net = CharRNN(use_cuda, vocab_size, hidden_size, layers, embed_size)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
train_mod = train[:len(train) - len(train)%128]
saved_train = []
total_loss = []
for epoch in tqdm(range(epochs)):
    if epoch % (epochs / 10) == 0:
        save_state(net, epoch)
    x, y = getData(train, sentence_len, batch_size)
    x, y = x.add(-1), y.add(-1)
    hidden = net.init_hidden(batch_size)
    optimizer.zero_grad()
    loss = 0
    for charIdx in range(sentence_len):
        input, label = x[:, charIdx], y[:, charIdx]
        output, hidden = net(input, hidden)
        # print('Criterion - Output: ', output.size(), ' Label: ', label.size())
        output = output.view(batch_size, -1)
        loss += criterion(output, label)
    total_loss.append(loss.data.numpy())
    loss.backward()
    optimizer.step()
plt.plot(range(epochs), total_loss, marker=(4, 0))
plt.xlabel('epoch')
plt.ylabel('NLL (lower is better)')
plt.savefig('Loss_Plot.jpg')
print('Saving State to laststate.pt')
net.save_state_dict('laststate.pt')

