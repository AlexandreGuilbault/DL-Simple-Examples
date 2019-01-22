import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, utils
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

#####################
# Settings
batch_size = 64
n_inputs = 28
n_outputs = 10
n_color_channels = 1

n_epochs = 10
dropout_probability = 0.5
learning_rate = 0.01
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
class SimpleCNN(nn.Module):
    def __init__(self, n_color_channels, n_inputs, n_outputs):
        super(SimpleCNN, self).__init__()
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_color_channels = n_color_channels
        
        # CNN output_size = (input_size - kernel_size + 2*padding)/stride_size + 1
        
        # 1x28x28
        self.conv = nn.Conv2d(self.n_color_channels, 8, kernel_size=4, stride=2, padding=1) 
        # 8x14x14
        self.ll = nn.Linear(self.n_color_channels*self.n_inputs**2*2, self.n_outputs) 
        self.dropout = nn.Dropout(dropout_probability)
        
    def forward(self, X):
 
        X = F.relu(self.conv(X.view(-1, 1, self.n_inputs, self.n_inputs)))
        X = self.dropout(X)
        X = self.ll(X.view(-1, self.n_color_channels*self.n_inputs**2*2))
        
        return X.view(-1, self.n_outputs)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(n_color_channels, n_inputs, n_outputs).to(device)

############################
# Test Model Before Training
dataiter = iter(trainloader)
images, labels = dataiter.next()
output = model(images)
print(output[0:10])
##################################


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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