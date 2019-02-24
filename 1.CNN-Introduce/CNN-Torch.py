'''
    code by Tae Hwan Jung @graykode
'''
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def img_show(image):
    image = image / 2 + 0.5
    if image.shape[0] == 3:
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    elif image.shape[0] == 1:
        plt.imshow(image.squeeze(0))
    plt.show(block=False)

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root='../data/', transform=transform)
cat, dog = dataset[0][0], dataset[1][0]

# What is filter(=kernel) in CNN
print('original image')
img_show(cat)

print('in_channels=3, out_channels=6, kernel_size=4 Convolution')
outputs = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4)(cat.unsqueeze(0)).data
for i in range(outputs.shape[1]):
    print(i+1,'channel')
    img_show(outputs[:,i,:,:])

print('in_channels=3, out_channels=6, kernel_size=40 Convolution')
outputs = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=40)(cat.unsqueeze(0)).data
for i in range(outputs.shape[1]):
    print(i+1,'channel')
    img_show(outputs[:,i,:,:])

print('in_channels=3, out_channels=6, kernel_size=3 stride=4 Convolution')
outputs = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=4)(cat.unsqueeze(0)).data
for i in range(outputs.shape[1]):
    print(i+1,'channel')
    img_show(outputs[:,i,:,:])