#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Building off of the following article:
#https://blog.paperspace.com/autoencoder-image-compression-keras/ 

#The following changes were made:
# 1) A convolutional layer was added to the model in the first/last step of the encoder/decoder
# 2) A ReLU and TanH activation were added to the end of the decoder.
# 3) A parameter for the "stored" (value between encoder & decoder) was added to the model
# 4) Results are now shown after 1 epoch and 5 epochs of training

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#1 layer CNN with output activation of ReLu and Tanh 
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        stored = 16
        #1 Encoder 
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1,8,5),
            nn.LeakyReLU()
        )
        self.pool = nn.MaxPool2d(2,stride=2,return_indices=True)
        
        self.encoder_linear = nn.Sequential(
            nn.Linear(in_features=1152, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=stored),
            nn.LeakyReLU()
        )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(in_features=stored, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256,out_features=1152),
            nn.LeakyReLU(),
        )
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        
        self.decoder_conv2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8,1,5),
            nn.ReLU(),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder_conv1(x)
        x, indices = self.pool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.encoder_linear(x)
        x = self.decoder_linear(x)
        x = torch.reshape(x, (-1,8,12,12))
        x = self.unpool(x,indices)
        x = self.decoder_conv2(x)
        return x


# In[3]:


#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AE(input_shape=784).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.05)

#MSE Loss
criterion = nn.MSELoss()


# In[4]:


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4
)


# In[5]:


for i in range(1,6,4):
    #Create a new model for test of epochs (reset the model)
    model = AE(input_shape=784).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Running for %d epochs" % i)
    for epoch in range(i):
        loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(inputs)
            train_loss = criterion(outputs, inputs)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}, loss = {:.6f}".format(epoch + 1,loss))
    get_ipython().run_line_magic('matplotlib', 'inline')
    test_examples = None

    with torch.no_grad():
        for i, data in enumerate(test_loader,0):
            #batch_features = batch_features[0]
            test_examples, test_labels = data
            #test_examples = batch_features.view(-1, 784)
            reconstruction = model(test_examples)
            break
    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].numpy().reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

