# ChatBot
This project is a chatbot suited for an "office environment". The goal is for it to help users to run various tasks like general questions, showing sales, typical conversation, or employee/ company information. It uses a feed-forward neural network made with PyTorch to have a contextual conversation.

### Approach:
1. NLP Processing (stemming, tokenization, create bag of words to reference to)
2. Create the training data / file for the model
3. Feed Forward Neural Network with CrossEntropyLoss
4. Using saved, trained model, implement the chat

### Tools Used:
- PyTorch
- JSON
- NLTK
- Numpy

### Final Result:
[![Screen-Shot-2020-12-02-at-1-39-00-AM.png](https://i.postimg.cc/j2wdDnLw/Screen-Shot-2020-12-02-at-1-39-00-AM.png)](https://postimg.cc/hQnqwj8c)

### To Do List
- [ ] Use Microsoft API to implement speech-to-text and vice versa. 
- [ ] Add more intents to train the model and make it more accurate
- [ ] Implement model with arduino robot to perform tasks like open websites, databases, and more on host computer. 


