'''
    code by Tae Hwan Jung @graykode
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1 * 256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, 10)

    def forward(self, x): # x : [1, 3, 227, 227]
        out = F.relu(self.conv1(x)) # out : [1, 96, 55, 55]
        out = F.max_pool2d(out, kernel_size=3, stride=2) # out : [1, 96, 27, 27]
        out = F.normalize(out)

        out = F.relu(self.conv2(out)) # out : [1, 256, 27, 27]
        out = F.max_pool2d(out, kernel_size=3, stride=2) # out : [1, 256, 13, 13]
        out = F.normalize(out)

        out = F.relu(self.conv3(out)) # out : [1, 384, 13, 13]
        out = F.relu(self.conv4(out)) # out : [1, 384, 13, 13]
        out = F.relu(self.conv5(out)) # out : [1, 256, 13, 13]
        out = F.max_pool2d(out, kernel_size=3, stride=2) # out : [1, 256, 6, 6]
        out = F.dropout(out)

        out = out.view(out.size(0), 256 * 6 * 6)
        out = F.relu(self.fc1(out)) # out : [1, 4096]
        out = F.dropout(out)
        out = F.relu(self.fc2(out)) # out : [1, 4096]

        out = self.classifier(out) # out : [1, 10]
        return out

batch_size = 1000
transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])
train_data = datasets.CIFAR10(root='./data/', train=False, transform=transform, download=True) # 10000 datas
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

model = AlexNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(model)
for epoch in range(10):
    # Train each batch(=1000), 10 times loop(=10000/1000)
    total_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    #print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss))