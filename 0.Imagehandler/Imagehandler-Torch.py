# code by Tae Hwan Jung(Jeff Jung) @graykode
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable

# image from https://github.com/ardamavi/Dog-Cat-Classifier/tree/master/Data/Train_Data
# 0 : cat, 1 : dog

def img_show(image):
    image = image / 2 + 0.5
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.show(block=False)

def pick_image(data, index):
    image, target = data[index]
    img_show(image)

original_image = datasets.ImageFolder(root='../data/', transform=transforms.ToTensor())
print(original_image,'\n')
pick_image(original_image, 1)

# make transformation (resizing image)
resized_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])
resized_image = datasets.ImageFolder(root='../data/', transform=resized_transform)
print('resized image to 227x227x3')
pick_image(resized_image, 1)

# make transformation (crop image)
cropped_transform = transforms.Compose([
    transforms.CenterCrop((10, 10)),
    transforms.ToTensor()
])
cropped_image = datasets.ImageFolder(root='../data/', transform=cropped_transform)
print('cropped image to 10x10x3')
pick_image(cropped_image, 1)

# make transformation (normalized image)
normalized_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

normalized_image = datasets.ImageFolder(root='../data/', transform=normalized_transform)
print('normalized image to mean and std (0.5, 0.5, 0.5)')
pick_image(normalized_image, 1)


# How to use Data Loader (Input Pipeline)
# same batch images should have same height, weight, channel
batch_size = 2
print('Data Loader')
dataloader = torch.utils.data.DataLoader(dataset=resized_image, batch_size=batch_size, shuffle=True)

count = 0
for batch_idx, (data, target) in enumerate(dataloader):
    data, target = Variable(data), Variable(target)
    count += batch_size
    print('batch :', batch_idx + 1,'    ', count, '/', len(original_image),
          'image:', data.shape, 'target : ', target)