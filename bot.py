import random
import json
import torch
from toolkit import bagOfWords, tokenize
from model import Net

with open('intents.json', 'r') as f:
    intents = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#LOAD MODL
PATH = "data.pt"
path = torch.load(PATH)

#PARAMS FROM MODEL
inputSize = path["input_size"]
hiddenSize = path["hidden_size"]
outputSize = path["output_size"]
all_words = path['all_words']
tags = path['tags']
model_state = path["model_state"]

model = Net(inputSize, hiddenSize, outputSize).to(device)
model.load_state_dict(model_state)
model.eval()

#CHAT
name = "FGF-BOT"

while True:
    user = input("YOU: ")

    user = tokenize(user)
    pog = bagOfWords(user, all_words) #Pog shape first index now
    pog = pog.reshape(1, pog.shape[0])
    pog = torch.from_numpy(pog).to(device)

    output = model(pog)
    #print(output)
    _, predicted = torch.max(output, dim=1)
   # print(predicted)
    tag = tags[predicted.item()]
    #print(tag)

    probs = torch.softmax(output, dim=1)
    probability = probs[0][predicted.item()]
    #print(probability)

    #to do: Could use argmax to find highest prob like MNIST

    #OUTPUT
    if probability.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                choice = random.choice(intent['responses'])
                print(name,": " , choice)

    else:
        print(name, "i dont understand...")


