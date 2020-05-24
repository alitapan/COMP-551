import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
import tensorflow as tf

# Convert images to tensors (multidimensional matrices)
# and normalize images in the range [-1,1].
transform = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 0.5 is the standard deviation

# Load the CIFAR10 dataset into training and test set
trainset = torchvision.datasets.CIFAR10(root='.\data', train=True,
                                        download=True, transform=transform)
# 4 samples per batch, 2 subprocesses to use for data-loading.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='.\data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
# The classes for the prediction task
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to show image
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # swap color axis because numpy image: H x W x C and torch image: C x H x W
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Neural network class that takes 3-channel images as inputs
# modified from the NN base class

# Net class has two paramters:
#       - mode: we have 5 modes as described in the report for trying different configurations and comparing their accuracies and loss
#       - loss: we have 2 different loss functions, loss = 1 is Cross-Entropy while loss = 2 is Negative Log-Likelihood

class Net(nn.Module):
    def __init__(self, mode, loss):

        super(Net, self).__init__()
        self.mode = mode
        self.loss = loss

        if(mode == 1):  # Control - Same as tutorial
          self.conv1 = nn.Conv2d(3, 6, 5)
          self.conv2 = nn.Conv2d(6, 16, 5)
          self.fc1 = nn.Linear(16 * 5 * 5, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, 10)

        if(mode == 2): # Extra convolutional layer
          self.conv1 = nn.Conv2d(3, 6, 4)
          self.conv2 = nn.Conv2d(6, 16, 4)
          self.conv3 = nn.Conv2d(16, 128, 4)
          self.fc1 = nn.Linear(128, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, 10)

        if(mode == 3):  # Extra linear layer
          self.conv1 = nn.Conv2d(3, 6, 5)
          self.conv2 = nn.Conv2d(6, 16, 5)
          self.fc1 = nn.Linear(16 * 5 * 5, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, 84)
          self.fc4 = nn.Linear(84, 10)

        if(mode == 4):  # Increased input and output
          self.conv1 = nn.Conv2d(3, 128, 10)
          self.conv2 = nn.Conv2d(128, 128, 10)
          self.fc1 = nn.Linear(128, 512)
          self.fc2 = nn.Linear(512, 1024)
          self.fc3 = nn.Linear(1024, 10)

        if(mode == 5):  # Increased input and output + extra convolutional layer
          self.conv1 = nn.Conv2d(3, 128, 4)
          self.conv2 = nn.Conv2d(128, 512, 4)
          self.conv3 = nn.Conv2d(512, 128, 4)
          self.fc1 = nn.Linear(128, 512)
          self.fc2 = nn.Linear(512, 1024)
          self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):

        if(self.mode == 1):  # Control - Same as tutorial
          x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
          x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
          x = x.view(-1, 16 * 5 * 5)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)

        if(self.mode == 2):  # Extra convolutional layer
          x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
          x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
          x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
          x = x.view(-1, 128)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)

        if(self.mode == 3): # Extra linear layer
          x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
          x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
          x = x.view(-1, 16 * 5 * 5)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = F.relu(self.fc3(x))
          x = self.fc4(x)

        if(self.mode == 4):  # Increased input and output
          x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
          x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
          x = x.view(-1, 128)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)

        if(self.mode == 5):  # Increased input and output + extra convolutional layer
          x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
          x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
          x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
          x = x.view(-1, 128)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)

        # Change return for loss function
        if(self.loss == 1):
          return x # For Cross Entropy loss function
        if(self.loss == 2):
          return F.log_softmax(x, dim=1) # For Negative Log-Likelihood loss function

import torch.optim as optim

net = Net(5, 2)
EPOCH = 5 # Number of loops to go over the data for training

criterion = nn.CrossEntropyLoss()

# Use stochastic gradient descent with momentum
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training the network
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        if(net.loss == 1):
          loss = criterion(outputs, labels)  # Cross-entropy Loss
        elif(net.loss == 2):
          loss = F.nll_loss(outputs, labels)  # Negative Log-Likelihood

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():  # do not calculate gradient, just checking accuracy
    # loop over data, count correct outputs
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # prediction = class with the highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# accuracy for each classs
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
