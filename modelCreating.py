"""
@Author: Paula Andrea Pérez Toro
@Date:   2018-11-29T18:13:18-05:00
@Email:  paulaperezt16@gmail.com
# @Last modified by:   Paula Andrea Pérez Toro
# @Last modified time: 2018-11-29T18:40:20-05:00

"""

from time import time
start_time = time()
print(start_time)

from NLP_FeatureExtraction import LDADictionary, Word2VecTraining


#########Creating Models#################

#
LDADictionary=LDADictionary()
LDADictionary.save("D:/Gita/GITA_Master/NLPCodes/Models/esLDADictionary.dict")
#To read it
#dictionary = corpora.Dictionary.load("D:/Gita/GITA_Master/NLPCodes/Models/estestDict.dict")

Word2VecModel=Word2VecTraining()
Word2VecModel.save("D:/Gita/GITA_Master/NLPCodes/Models/esWord2Vec.model")
#To read it
#Word2Vec.load(D:/Gita/GITA_Master/NLPCodes/Models/esWord2Vec.model")

elapsed_time = time() - start_time
print("Elapsed time: %0.10f seconds." % elapsed_time)
