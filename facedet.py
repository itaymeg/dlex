import torch
import torchfile
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pascal as voc
from nms import NMS

def caten_datasets(arr1, arr2):
    result = []
    for _, data in enumerate(arr1):
        result.append(data)
    for _, data in enumerate(arr2):
        result.append(data)
    return np.asarray(result)


# load data using torchfile
aflw12 = torchfile.load('./aflw/aflw_12.t7')
aflw24 = torchfile.load('./aflw/aflw_24.t7')
aflw12 = np.array(list(aflw12.values()), dtype=np.double)
aflw24 = np.array(list(aflw24.values()), dtype=np.double)
aflw12_labels = np.ones(len(aflw12))
aflw24_labels = np.ones(len(aflw24))
pascal = voc.prepare_data_facedet(crops_per_image=3)
pascal_labels = np.zeros(len(pascal)) #we take images from pascal that are not faces
#catenate aflw and pascal for dataset
x_12 = caten_datasets(aflw12, pascal)
y_12 = caten_datasets(aflw12_labels, pascal_labels)
dset12 = TensorDataset(torch.from_numpy(x_12), torch.from_numpy(y_12))
dset24 = TensorDataset(torch.from_numpy(aflw24), torch.ones(len(aflw24)))
dload12 = DataLoader(dset12, batch_size=32, shuffle=True, num_workers=8)
dload24 = DataLoader(dset24, batch_size=32, shuffle=True, num_workers=8)

# {id: 3 x 12 x 12}

class Net12(nn.Module):
    def __init__(self):
        super(Net12,self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=(3,3), stride=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.fc = torch.nn.Linear(4 * 4 * 16, 2)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

net = Net12().double()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
stat = []
#train
for epoch in range(10):
    for i,data in enumerate(dload12):
        inputs, labels = data
        inputs, labels = Variable(inputs).double(), Variable(labels).long()
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, labels)
        if i == 0:
            print("loss= " ,loss.data.numpy(), " epoch: ", epoch)
        stat.append(loss.data.numpy())
        loss.backward()
        optimizer.step()
plt.plot(stat)
plt.show()



