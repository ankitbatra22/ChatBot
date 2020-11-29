import nltk
#nltk.downlod('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

"""tokenize
lower and stem
take out punctiation
create bag of words"""

def tokenize(sentance):
    return nltk.word_tokenize(sentance)

def stem(word):
    return stemmer.stem(word.lower())

def bagOfWords(tokenized_sentance, all):
    pass

a = "how long does shipping take?"
#print(a)
print(tokenize(a))