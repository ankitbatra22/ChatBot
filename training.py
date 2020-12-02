import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from toolkit import tokenize, stem, bagOfWords
from model import Net

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
arr = []

# loop through each sentence in the intents (json)
for i in intents['intents']:
    tag = i['tag']
    # create tag array
    tags.append(tag)
    for pattern in i['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to the words array
        all_words.extend(w)
        # create xy pair for trainind
        arr.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create bag of words, TRY SEPERATING FROM

xTrain = []
yTrain = []

for (patternSentance, tag) in arr:
    bog = bagOfWords(patternSentance, all_words)
    xTrain.append(bog)
    label = tags.index(tag)
    yTrain.append(label)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)


# DATASET

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(xTrain)
        self.x_data = xTrain
        self.y_data = yTrain

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


###TRAINING PIPELINE ###

# PARAMS
batchSize = 8
hiddenSize = 8
outputSize = len(tags)
inputSize = len(xTrain[0])
learningRate = 0.001
epochs = 1500

# Model
dataset = ChatDataset()
trainLoader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0) #NUM WORKERS TRY 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(inputSize, hiddenSize, outputSize).to(device)

# loss / optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)


for epoch in range(epochs):
    for words,labels in trainLoader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(words)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # calculate the loss
        loss = criterion(output, labels)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": inputSize,
"hidden_size": hiddenSize,
"output_size": outputSize,
"all_words": all_words,
"tags": tags
}

PATH = "data.pt"
torch.save(data, PATH)

print("done training :)")
