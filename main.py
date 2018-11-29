"""
@Author: Paula Andrea Pérez Toro
@Date:   2018-11-28T20:59:24-05:00
@Email:  paulaperezt16@gmail.com
# @Last modified by:   Paula Andrea Pérez Toro
# @Last modified time: 2018-11-28T21:55:03-05:00

"""
from NLP_PreProcessing import noPunctuation, StopWordsRemoval, Lemmatizer
from NLP_FeatureExtraction import Bag_of_WordsFT, TF_IDF, LDA, Word2Vec

#########Testing Functions#################
f = open ('D:/Gita/GITA_Master/NLPCodes/030YHC_S1_monologue.txt','r')
text = f.read()

f.close()

text2='Yo claro que tenía una vida, y mantengo un futuro y me quiero mucho'
print('%%%%%%%%%%%Texto Normal%%%%%%%%%%%%%%%%%')
print(text)
ProcessingText1=noPunctuation(text)
ProcessingText2=noPunctuation(text2)
print('%%%%%%%%%%%Sin puntuacion%%%%%%%%%%%%%%%%%')
print(ProcessingText1)
#print(ProcessingText2)
ProcessingText21=StopWordsRemoval(ProcessingText1)
print('%%%%%%%%%%%Sin stopwords%%%%%%%%%%%%%%%%%')
print(ProcessingText21)
ProcessingText22=StopWordsRemoval(ProcessingText2)
#print(ProcessingText22)
ProcessingText31=Lemmatizer(ProcessingText21)
print('%%%%%%%%%%%Lematizado%%%%%%%%%%%%%%%%%')
print(ProcessingText31)

ProcessingText32=Lemmatizer(ProcessingText22)
#print(ProcessingText32)
docs=[ProcessingText31,ProcessingText32]

FTBoW=Bag_of_WordsFT(docs)
print('%%%%%%%%%%%Bag of words%%%%%%%%%%%%%%%%%')
print(FTBoW)


FTTF_IDF=TF_IDF(docs)
print('%%%%%%%%%%%TF-IDF%%%%%%%%%%%%%%%%%')
print(FTTF_IDF)



FTLDA=LDA(docs)
print(FTLDA)

FTWord2Vec=LDA(docs)
print(FTWord2Vec)
