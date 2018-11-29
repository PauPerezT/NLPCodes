"""
@Author: Paula Andrea Pérez Toro
@Date:   2018-11-28T20:59:24-05:00
@Email:  paulaperezt16@gmail.com
# @Last modified by:   Paula Andrea Pérez Toro
# @Last modified time: 2018-11-28T23:23:12-05:00

"""
from NLP_PreProcessing import noPunctuation, StopWordsRemoval, Lemmatizer
from NLP_FeatureExtraction import Bag_of_WordsFT, TF_IDF, LDADictionary, LDA, Word2VecTraining, Word2VecFTE
from os import listdir
#########Testing Functions#################

path="D:/Gita/GITA_Master/NLPCodes/txt_emotion_test/"
docs=[]
for file in listdir(path):
    f = open (path+file,'r')
    text = f.read()
    ProcessingText_NP=noPunctuation(text)
    ProcessingText_SW=StopWordsRemoval(ProcessingText_NP)
    ProcessingText_LMM=Lemmatizer(ProcessingText_SW)
    docs.append(ProcessingText_LMM)

f.close()



FTBoW=Bag_of_WordsFT(docs)
print('%%%%%%%%%%%Bag of words%%%%%%%%%%%%%%%%%')
print(FTBoW)


FTTF_IDF=TF_IDF(docs)
print('%%%%%%%%%%%TF-IDF%%%%%%%%%%%%%%%%%')
print(FTTF_IDF)



LDADictionary=LDADictionary()
FTLDA=LDA(texts=docs,dictionary=LDADictionary)
print(FTLDA)

Word2VecModel=Word2VecTraining()
FTWord2Vec=Word2VecFTE(docs,Word2VecModel)
print(FTWord2Vec)
