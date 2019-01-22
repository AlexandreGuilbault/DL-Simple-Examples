# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:11:56 2018

@author: Alexandre Guilbault
"""
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, utils
import torch.nn.functional as F
import torch.optim as optim

#####################
# Settings
batch_size = 64
n_inputs = 28*28
n_outputs = 10
n_epochs = 10
#####################


#####################
# Dataset
transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.MNIST(root='./data', train=True,download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

testloader = utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
trainloader = utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)


#####################
# Functions
def calculate_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    return (corrects/batch_size).item()


#####################
# Verify Dataset
dataiter = iter(trainloader)
images, labels = dataiter.next()

images = torchvision.utils.make_grid(images).numpy()
plt.imshow(np.transpose(images, (1, 2, 0)))


#####################
# Define Model
class SimpleMLP(nn.Module):
    def __init__(self, batch_size, n_inputs, n_outputs):
        super(SimpleMLP, self).__init__()
        
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
            
        self.l1 = nn.Linear(self.n_inputs, self.n_inputs*2) 
        self.l2 = nn.Linear(self.n_inputs*2, self.n_inputs)
        self.l3 = nn.Linear(self.n_inputs, self.n_outputs)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, X):
        X = F.relu(self.l1(X.view(-1,self.n_inputs)))
        X = self.dropout(X)
        X = F.relu(self.l2(X))
        X = self.dropout(X)
        X = self.l3(X)
        
        return X.view(-1, self.n_outputs)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleMLP(batch_size, n_inputs, n_outputs).to(device)

############################
# Test Model Before Training
dataiter = iter(trainloader)
images, labels = dataiter.next()
output = model(images)
#print(output[0:10])
##################################


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, n_epochs+1):
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()
    
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += calculate_accuracy(outputs, labels, batch_size)
         
    print('Epoch: {} | Loss: {:.4} | Train Accuracy: {:.1%}'.format(epoch, train_running_loss/i, train_acc/i))


model.eval()
test_acc = 0.0
for i, data in enumerate(testloader, 0):
    
    inputs, labels = data
    inputs = inputs.view(-1, 28, 28)
    
    outputs = model(inputs)
    test_acc += calculate_accuracy(outputs, labels, batch_size)
        
print('Test Accuracy: {:.1%}'.format(test_acc/i))