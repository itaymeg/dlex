## -*- coding: utf-8 -*-
#"""
#Created on Mon Nov 13 16:36:49 2017
#
#@author: Art
#"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import mode
import random
from itertools import groupby


#import random
#import tarfile
import inspect
from sklearn.utils import shuffle

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.serialization import load_lua
torch.manual_seed(42)





## loading data and making small subset
## vocab.t7 and train.t7 files should be in same folder with the code
vocab_charToIdx = load_lua('vocab.t7')
vocab_IdxToChar = dict(zip(vocab_charToIdx.values(), vocab_charToIdx.keys()))
full_train_byte_tensor = load_lua('train.t7')
print("Number of characters in train data:")
print(full_train_byte_tensor.size())
small_train_byte_tensor = full_train_byte_tensor

##printing the small subset of data in form of characters
#first_characters_in_small_tensor = [vocab_IdxToChar[letter] for letter \
#                                    in small_train_byte_tensor]
#print("Small train data translated to characters:")
#print(''.join(first_characters_in_small_tensor))



#########################################################################
#########################################################################
###  Wasted a few hours loading the data for "word"-level RNN         ###
###  only to understand that we need "character"-level RNN            ###
###  we can use it to compare between char-level and word-level RNN   ###
#########################################################################
#########################################################################
#
### input character byte tensor representing training data and list of poncuation
### output list of list of "int" representing separate words and punctuation (including space)
#def split_byte_tensor_to_wordByteList(byteTensorWordData, punct_idx_list):
#    return [list(group) for k, group in groupby(list(byteTensorWordData),\
#                 lambda x: x in punct_idx_list)]
### input character byte tensor representing training data and vocabulary
### output list of list of "int" representing separate words and punctuation (without space)   
#def split_byte_tensor_to_punctuation_wordByteList(byteTensorWordData,vocab_charToIdx):
#    ## setting the punctuation lists
#    punct_char_list = [" ", '!', '(', ')', ',', '.', ':', ';', '?']
##    notPunct_char_list = [char for char in vocab_charToIdx if char not in punct_char_list]
#    punct_idx_list = [vocab_charToIdx[letter] for letter in punct_char_list]
##    notPunct_idx_list = [vocab_charToIdx[letter] for letter in notPunct_char_list]
#    
#    wordByteList = split_byte_tensor_to_wordByteList (byteTensorWordData,punct_idx_list)
#    wordByteList_noSpace = list(map(lambda x: [k for k in x if k != 1], wordByteList))
#    return list(filter(None, wordByteList_noSpace))
#    
### input list of list of "int" representing separate words and punctuation (wordByteList)
### outout - None
### prints the words represented in the output
#def print_wordByteList(wordByteList, vocab_IdxToChar):
#    for word in wordByteList:
#        print(''.join([vocab_IdxToChar[letter] for letter in word]))
#
#
### test all data loading is done
### should print list of words and punctuations (without spaces) of the small training subset
##print_wordByteList(split_byte_tensor_to_punctuation_wordByteList\
##                   (small_train_byte_tensor,vocab_charToIdx),\
##                   vocab_IdxToChar) 
#
#
#temp_sentence = split_byte_tensor_to_punctuation_wordByteList\
#                   (small_train_byte_tensor,vocab_charToIdx)
#temp_sentence = [tuple(row) for row in temp_sentence]
#
## build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
#trigrams = [([temp_sentence[i], temp_sentence[i + 1]], temp_sentence[i + 2])
#            for i in range(len(temp_sentence) - 2)]
## print the first 3, just so you can see what they look like
##print(trigrams[:3])
#
#vocab = set(temp_sentence)
#word_to_ix = {word: i for i, word in enumerate(vocab)}
#temp_sentence_idx = [word_to_ix[w] for w in temp_sentence]
#wordIdxByteLongTensor = torch.ByteTensor(temp_sentence_idx)
#


#########################################################################
#########################################################################
###  Wasted a few hours loading the data for "word"-level RNN         ###
###  only to understand that we need "character"-level RNN            ###
###  we can use it to compare between char-level and word-level RNN   ###
#########################################################################
#########################################################################









## data loader for model
def charDataLoader(train_tensor, sentence_len, batch_size):
    first_char = torch.LongTensor(batch_size, sentence_len)
    second_char = torch.LongTensor(batch_size, sentence_len)
    for batch_idx in range(batch_size):
        start_index = random.randint(0, len(train_tensor) - sentence_len - 1)
        end_index = start_index + sentence_len + 1
        sentence = train_tensor[start_index:end_index]
        first_char[batch_idx] = sentence[:-1]
        second_char[batch_idx] = sentence[1:]
    return Variable(first_char), Variable(second_char)

## the model
class Ex3_charNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, run_cuda, layer_num):
        super(Ex3_charNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.run_cuda = run_cuda
        self.layer_num = layer_num

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, self.layer_num)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        embeded_input = self.embed(input)
        output, hidden = self.lstm(embeded_input.view(1, batch_size, -1), hidden)
        output = self.linear(output.view(batch_size, -1))
        output = F.softmax(output)
        return output, hidden



    def init_hidden(self, batch_size):
        if self.run_cuda:
            return (Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size)).cuda())
        return (Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size)))
        
def generate(model, initial_seed=None, predict_len=100, temperature=0.8, \
             run_cuda=True, vocab_charToIdx = vocab_charToIdx, vocab_IdxToChar = vocab_IdxToChar):
    model.eval()
    if initial_seed == None:
        initial_seed = [random.choice(list(vocab_charToIdx.values()))-1]
    hidden = model.init_hidden(1)
    initial_input = Variable(torch.LongTensor(initial_seed).unsqueeze(0))

    if run_cuda:
        initial_input = initial_input.cuda()
    
    predicted = []
    for char in initial_seed:
        predicted.append(vocab_IdxToChar[char+1])


    # Use initial seed to "build up" hidden state
    for p in range(len(initial_seed) ):
        _, hidden = model(initial_input[:,p], hidden)
        
    input_char = initial_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = model(input_char, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
#        print(top_i)
        predicted_char = vocab_IdxToChar[top_i+1]
        predicted.append(predicted_char)
        input_char = Variable(torch.LongTensor([top_i]).unsqueeze(0))
        if run_cuda:
            input_char = input_char.cuda()
    model.train()
    return predicted        

## parameters
train_tensor = small_train_byte_tensor.long()
sentence_len = 100
batch_size = 256
num_epochs = 10000
voc_size = len(vocab_charToIdx)
hidden_size = 512
layer_num = 1
learning_rate = 0.01
run_cuda = False
all_losses = []
loss_avg = 0


model = Ex3_charNN(
    voc_size,
    hidden_size,
    voc_size,
    run_cuda,
    layer_num
)
if run_cuda:
    model.cuda()
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()




print("Training for %d epochs..." % num_epochs)
for epoch in range(num_epochs):
    inputs, targets = charDataLoader(train_tensor, sentence_len, batch_size)
    ## important - lowering the indexes by one
    ## because the stupid dataset given to us starts it's indexing from "1" and not "0"
    inputs, targets = inputs.add(-1), targets.add(-1)
    if run_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    
    hidden = model.init_hidden(batch_size)

    model.zero_grad()
    loss = 0

    for character_idx in range(sentence_len):
        output, hidden = model(inputs[:,character_idx], hidden)
        loss += criterion(output.view(batch_size, -1), targets[:,character_idx])

    loss.backward()
    optimizer.step()

    all_losses.append( loss.data[0] / sentence_len)


    if epoch % 100 == 0:
        learning_rate = learning_rate/2
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if epoch % 10 == 0:
        print('[ (%d %d%%) %.4f]' % ( epoch, epoch / num_epochs * 100, all_losses[-1]))
    if epoch % 50 == 0:
        print(''.join(generate(model, initial_seed=None, predict_len=70, temperature=0.18)), '\n')

plt.plot(all_losses)
    
    
## The code will train and generate samples
## it requires ~5GB of memory (ram if on CPU and GPU memory if with cuda) with the current parameters
## I didn't test it on CPU

## On my PC the code takes about an hour to train for 10000 epochs
## in this case, epoch is not when the code trains on the whole dataset once...
## one epoch in this case is after the code trains on "batch_size" number of "sentences"
## the sentences are just random snippets of strings from the whole dataset 
## and each sentence is of length "sentence_len"
## you can change all those settings in the parameters section

## Some strategies and facts about the code:
## - The longer the sentence, the longer is the "context"
##      so if we will take sentence lengths of 10-20 then the NN will learn
##      simple words and spelling, but will not know about parenthesis and that
##      you need to open and close them, because in 10-20 characters there won't 
##      be any whole set of parenthesis
##      On the same thought, if you give it long sentences, then the NN might learn 
##      actual sentence structure, words following words etc...
## - The number of hidden layers is the number of "embeddings" I think more
##      will give the NN more ability to learn connections and context awareness
##      but I'm not sure about that. You should experiment with different numbers
## - The layer number is currently 1 but you might increase it to 2/3 in order
##      to make the NN to understand better the context throught time...
##      I read somewhere that more than 2/3 layers is non beneficial...
## - I'm not sure if batch size is effecting the learning... 
##      you should try different sizes, maybe it will make the NN more generalized
## - The biggest hits to the amout of memory required are in order:
##      sentence_len, num_layers, hidden_size, batch_size.

## Now about generation:
## the temperature value is controling how much "random" is present when selecting
## the next character to generate...
## so if the temperature is high (>0.5) then the for the next character we will 
## "have more probability to chose a random character" while if the temperature is low
## (<0.1) then the "we will almost certainly chose the character with the 
## highiest probability" 
## you should read the code and see what is the computation there...
## try generating code with different temperatures at different times of training
## to see what I'm talking about












