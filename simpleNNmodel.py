# Imports
import torch
from torch._C import device
import torch.nn as nn
import torch.optim as opt  # Adam,Stoch gradient descent, ...
import torch.nn.functional as F  # relu,tanh , ...
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Class for the Neural Network
class NN(nn.Module):
    def __init__(self, inpsize, no_of_classes):
        super(NN, self).__init__()
        # Input layer -> HiddenLayer1 -> HiddenLayer2 -> OutputLayer
        self.FC1 = nn.Linear(inpsize, 100)
        self.FC2 = nn.Linear(100, 75)
        self.FC3 = nn.Linear(75, no_of_classes)

    def forward(self, x):
        forward_vals = F.relu(self.FC1(x))
        forward_vals = torch.tanh(self.FC2(forward_vals))
        forward_vals = self.FC3(forward_vals)
        return forward_vals


# Hyperparameters
inpsize = 784
no_of_classes = 10
alpha = 0.005
batch_size = 64
num_epochs = 1

# Data loading
training_dataset = datasets.MNIST(
    root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=training_dataset,
                          batch_size=batch_size, shuffle=True)

testing_dataset = datasets.MNIST(
    root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=testing_dataset,
                         batch_size=batch_size, shuffle=True)

model = NN(inpsize, no_of_classes).to("cuda")

# Loss
loss = nn.CrossEntropyLoss()
# Optimizer
optimizer = opt.Adam(model.parameters(), lr=alpha)

# Training model
for epoch in range(num_epochs):
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device="cuda")
        label = label.to(device="cuda")

        data = data.reshape(data.shape[0], -1)

        scores = model(data)
        Loss = loss(scores, label)

        optimizer.zero_grad()
        Loss.backward()

        optimizer.step()


# Accuracy function
def find_accuracy(loader, model):
    correct = 0
    samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device="cuda")
            y = y.to(device="cuda")
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            correct += (predictions == y).sum()
            samples += predictions.size(0)

    acc = 0
    acc = (float(correct)/float(samples))*100

    model.train()
    return acc


train_accuracy = find_accuracy(train_loader, model)
test_accuracy = find_accuracy(test_loader, model)

print("training set accuracy=\t", train_accuracy)
print("test set accuracy=\t", test_accuracy)
