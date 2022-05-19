import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
"""
Stem the words in the sentence. This is used to reduce the size of the dictionary.
@param sentence - the sentence to stem           
@return the stemmed sentence
"""

import numpy
import tflearn
import tensorflow
import random
import json

with open('intents.json') as file:
    """
    Load the intents.json file and return the data.
    @param file - the file containing the intents.json data.
    @return the data from the file.
    """
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    """
    For each intent, tokenize the patterns and add them to the words list. Also, add the intent's tag to the labels list.
    @param data - the data dictionary
    @returns the words list and the labels list
    """

    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])
        
