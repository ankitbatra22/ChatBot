import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(Net, self).__init__()
        self.layerOne = nn.Linear(inputSize, hiddenSize)
        self.layerTwo = nn.Linear(hiddenSize, hiddenSize) # Feed Through
        self.layerThree = nn.Linear(hiddenSize, numClasses)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.layerOne(x)
        output = self.relu(output)
        output = self.layerTwo(output)
        output = self.relu(output)
        output = self.layerThree(output)

        return output

