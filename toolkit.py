import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()

"""
1. Tokenize (split sentance into array of words / tokens. This includes punctuation.
2. lower case and find root form of a word
3. take out punctuation
4. create bag of words. array with 1 for word that exists in a sentance and 0 for otherwise. 
"""

def tokenize(sentance):
    return nltk.word_tokenize(sentance)

def stem(word):
    return stemmer.stem(word.lower())

def bagOfWords(tokenized_sentance, all_words):
    tokenized_sentance = [stem(w) for w in tokenized_sentance]
    bagArray = np.zeros(len(all_words), np.float32)

    for index, w in enumerate(all_words):
        if w in tokenized_sentance:
            bagArray[index] = 1.0

    return bagArray






"""

IGNORE : TESTING

sentance = ["hello", "how", "are", "you"]
all = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag = bagOfWords(sentance, all)
print(bag)

a = "how long does shipping take?"
#a = (tokenize(a))
#print(a)

words = ["organize", "organizes", "organizing"]

stemmed_words = []
for w in words:
    stemmed = w
    stemmed_words.append(stem(stemmed))

print(stemmed_words)
"""

