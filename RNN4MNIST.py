import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, utils
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import time

#####################
# Settings
batch_size = 64
n_inputs = 28
seq_length = 28
n_outputs = 10

n_epochs = 10
dropout_probability = 0.5
learning_rate = 0.001
#####################


#####################
# Dataset
transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.MNIST(root='./data', train=True,download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

testloader = utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=0)
trainloader = utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)


#####################
# Functions
def calculate_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    return float(corrects)/batch_size


#####################
# Verify Dataset
dataiter = iter(trainloader)
images, labels = dataiter.next()

images = torchvision.utils.make_grid(images).numpy()
plt.imshow(np.transpose(images, (1, 2, 0)))


#####################
# Define Model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, seq_length, hidden_size, output_size, num_layers, dropout_probability):
        super(SimpleRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.num_layers = num_layers
        self.seq_length = seq_length

        if num_layers > 1:
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_probability, batch_first=True)
        else : # No dropout for last RNN layer
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.ll = nn.Linear(self.hidden_size*self.input_size, output_size)
        
    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))
        
    def forward(self, X):
        h = self.init_hidden(X.size(0))
        X = X.view(-1, self.seq_length, self.input_size)
        X, h = self.rnn(X, h) # X ->  (batch, seq, feature)
        
        X = self.ll(X.contiguous().view(X.size(0), -1))
        return X


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(input_size=n_inputs, seq_length=seq_length, hidden_size=128, output_size=n_outputs, num_layers=1, dropout_probability=dropout_probability).to(device)

############################
# Test Model Before Training

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# output = model(images)
# print(output[0:10])

##################################

start_time = time.time()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(1, n_epochs+1):
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()
    
    for i, data in enumerate(trainloader, 1):
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
training_time = time.time()-start_time

model.eval()
test_acc = 0.0
for i, data in enumerate(testloader, 1):
    
    inputs, labels = data
#    inputs = inputs.view(-1, 28, 28)
    
    outputs = model(inputs)
    test_acc += calculate_accuracy(outputs, labels, batch_size)

num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
print('\nRNN\n*******')
print('Number of parameters : {:,}\nTest accuracy: {:.1%}\nTraining time : {:.1f} seconds'.format(num_parameters, test_acc/i, training_time))
