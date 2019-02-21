# code by Tae Hwan Jung(Jeff Jung) @graykode
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def img_show(image, target):
    print('Class : ', classes[target])
    images = image / 2 + 0.5
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.show(block=False)

# About train option, True mean 50000 image datas, False 10000 image datas
original_dataset = datasets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
print(original_dataset,'\n')

image, target = original_dataset[7777] # pick index of 7777 data
img_show(image, target)

# make transformation (resizing image)
transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])
transformed_dataset = datasets.CIFAR10(root='./data/', train=False, transform=transform, download=True)
print(transformed_dataset,'\n')

image, target = transformed_dataset[7777] # pick index of 7777 data
img_show(image, target)

# How to use Data Loader (Input Pipeline)
batch_size = 1000
print('Data Loader')
train_loader = torch.utils.data.DataLoader(dataset=original_dataset, batch_size=batch_size, shuffle=True)

count = 0
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = Variable(data), Variable(target)
    count += batch_size
    print('batch :', batch_idx + 1, count, '/', len(original_dataset))