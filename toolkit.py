import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
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

def bagOfWords(tokenized_sentance, all):
    pass

"""a = "how long does shipping take?"
a = (tokenize(a))
print(a)

words = ["organize", "organizes", "organizing"]

stemmed_words = []
for w in words:
    stemmed = w
    stemmed_words.append(stem(stemmed))

print(stemmed_words)"""


