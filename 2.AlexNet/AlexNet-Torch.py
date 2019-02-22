'''
    code by Tae Hwan Jung @graykode
    reference : https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
'''
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms

def img_show(image):
    image = image / 2 + 0.5
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.show(block=False)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward(self, x): # x : [1, 3, 227, 227]
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root='../data/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=True)

model = AlexNet()
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(400):
    total_loss = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch + 1)%100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss))

# Test binary classification (cat is zero, dog is one)
for (data, target) in dataset:
    predict = model(data.unsqueeze(0).to(device))
    predict = predict.data.max(1, keepdim=False)[1].item()
    print('predict:','cat' if predict is 0 else 'dog', ' answer:', 'cat' if target is 0 else 'dog')