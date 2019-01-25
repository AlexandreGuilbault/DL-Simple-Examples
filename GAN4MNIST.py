import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, utils
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import time


# Some great ideas taken from here : https://github.com/soumith/ganhacks


#####################
# Settings
batch_size = 64
n_inputs = 28

n_epochs = 25
dropout_probability = 0.5
learning_rate = 0.002

z_size = 100
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

def real_loss(d_out, device='cpu', smooth=False):
    criterion = nn.BCEWithLogitsLoss()
    
    if smooth: loss = criterion(d_out.squeeze(), torch.ones(d_out.size(0)).to(device)*0.9)
    else: loss = criterion(nn.BCEWithLogitsLoss(d_out.squeeze(), torch.ones(d_out.size(0)).to(device)))
                                    
    return loss

def fake_loss(d_out, device='cpu', inverse=False):
    criterion = nn.BCEWithLogitsLoss()
    
    if inverse : loss = criterion(d_out.squeeze(), torch.ones(d_out.size(0)).to(device))
    else: loss = criterion(d_out.squeeze(), torch.zeros(d_out.size(0)).to(device))
    
    return loss

#####################
# Verify Dataset
dataiter = iter(trainloader)
images, labels = dataiter.next()

images = torchvision.utils.make_grid(images).numpy()
plt.imshow(np.transpose(images, (1, 2, 0)))


#####################
# Define Model
class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # output_size = (input_size - kernel_size + 2Padding)/stride + 1
        
        # 1x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=1, padding=0)
        # 8x25x25
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(16)
        # 16x12x12
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(32)
        # 32x5x5
        
        self.ll2 = nn.Linear(32*5*5, output_size)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x):

        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.lrelu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.ll2(x.view(-1,32*5*5))
    
        return x
    
    
class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size, dropout_probability=0.5):
        super(Generator, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.dropout_probability = dropout_probability

        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        
        self.ll1 = nn.Linear(input_size, 32*5*5)
        # 32x5x5
        self.convt1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(16) 
        # 16x11x11
        self.convt2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=0) 
        self.batch_norm2 = nn.BatchNorm2d(8) 
        # 8x13x13
        self.convt3 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=0) 
        # 4x28x28
        
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(self.dropout_probability)
        
    def forward(self, x):
        x = self.lrelu(self.ll1(x))
        x = self.lrelu(self.batch_norm1(self.convt1(x.view(-1,32,5,5))))
        x = self.dropout(x)
        x = self.lrelu(self.batch_norm2(self.convt2(x)))
        x = self.dropout(x)
        x = self.tanh(self.convt3(x))

        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
discriminator = Discriminator(n_inputs*n_inputs, n_inputs*n_inputs*2, 1).to(device)
generator = Generator(z_size, n_inputs*n_inputs*2, n_inputs*n_inputs).to(device)

############################
# Test Model Before Training
print(discriminator)
print()
print(generator)
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# output = model(images)
# print(output[0:10])

##################################

start_time = time.time() 

fixed_z = torch.randn(batch_size, z_size).to(device)

d_optimizer = optim.SGD(discriminator.parameters(),lr=learning_rate)
g_optimizer = optim.Adam(generator.parameters(),lr=learning_rate)

samples = []
losses = []
for epoch in range(1, n_epochs+1):
    train_running_loss = 0.0
    train_acc = 0.0
    discriminator.train()
    generator.train()
    
    for i, data in enumerate(trainloader,1):
        
    
        real_images, _ = data
        real_images = real_images.to(device)
        real_images = real_images*2-1  # Rescale input images from [0,1] to [-1, 1]


        #################################
        # Train Discriminator
        
        d_optimizer.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, z_size).to(device)
        fake_images = generator(z)
        
        # Discriminator with real images
        d_real = discriminator(real_images)
        r_loss = real_loss(d_real, device=device, smooth=True)

        # Discriminator with fake images
        d_fake = discriminator(fake_images)
        f_loss = fake_loss(d_fake, device=device)
        
        d_loss = r_loss + f_loss

        # Optimize Discriminator
        d_loss.backward()
        d_optimizer.step()
        
        #################################
        # Train Generator  
        
        g_optimizer.zero_grad()
        
        # Generate other fake images
        z = torch.randn(batch_size, z_size).to(device)
        fake_images = generator(z)
        
        d_fake = discriminator(fake_images)
        g_loss = fake_loss(d_fake, inverse=True, device=device)

        # Optimize Generator
        g_loss.backward()
        g_optimizer.step()       
        
        # Print losses
        if i % 200 == 0:
            print('Epoch {} | Batch_id {} | d_loss: {:.4f} | g_loss: {:.4f}'.format(epoch, i, d_loss.item(), g_loss.item()))

    # Generate and save samples of fake images
    losses.append((d_loss.item(), g_loss.item()))

    generator.eval() 
    fake_images = generator(fixed_z)
    samples.append(fake_images)


#######################
# Print training losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("GAN Training Losses")
plt.legend()

############################################
# Print generated fake images through epochs
fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
for ax, img in zip(axes.flatten(), samples[-25]):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    im = ax.imshow(img.detach().reshape((28,28)), cmap='Greys_r')
