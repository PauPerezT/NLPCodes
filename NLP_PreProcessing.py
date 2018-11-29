"""
@Author: Paula Andrea Pérez Toro <PauPerezT>
@Date:   2018-11-27T20:36:36-05:00
@Email:  paulaperezt16@gmail.com
@Filename: NLP_PreProcessing.py
# @Last modified by:   Paula Andrea Pérez Toro
# @Last modified time: 2018-11-28T18:15:46-05:00

"""
###English implementation in progress
"""
#%%%%%%% Natural Language Basic preprocessing code %%%%%%%#

    #%% Contains %%#
        *noPunctuation: to eliminate the Punctuation
        *StopWordsRemoval: to eliminate stopwords
        *Lemmatizer:

    #%% Variables %%#
        - Text: text to preprocess
        - Language: 'spanish' or 'english'

"""

#%% Import Libraries
import string
import spacy
from nltk.corpus import stopwords

#%% noPunctuation
def noPunctuation(text):
    #eliminate the punctuation in form of characters
    nopunctuation= [char for char in text if char not in string.punctuation]

    #Now eliminate the punctuation and convert into a whole sentence
    nopunctuation=''.join(nopunctuation)

    #Split each words present in the new sentence
    nopunctuation.split()

    return nopunctuation

#%% StopWordsRemoval
def StopWordsRemoval(text,language='spanish'):
    #Now eliminate stopwords
    clean_sentence= [word for word in text.split() if word.lower() not in stopwords.words('spanish')]

    return clean_sentence

#%% Lemmatizer
def Lemmatizer(text,language='spanish'):
    nlp = spacy.load('es_core_news_sm')
    doc = nlp(text)
    tokenLemma=[]

    for token in doc:
        print(token, token.lemma, token.lemma_)
        tokenLemma.append(token.lemma_)

    return tokenLemma
