"""
@Author: Paula Andrea Pérez Toro
@Date:   2018-11-28T11:41:08-05:00
@Email:  paulaperezt16@gmail.com
@Filename: NLP_FeatureExtraction.py
# @Last modified by:   Paula Andrea Pérez Toro
# @Last modified time: 2018-11-28T23:11:02-05:00

"""

###English implementation in progress
"""
#%%%%%%% Natural Language Basic Feature Extraction code %%%%%%%#

    #%% Contains %%#
        *Statistical Features:
            - Bag of Words
            - TF-IDF
        *Entity Extraction:
            - Topic Modeling-LDA
        *Word Emmbbedings:
            - Word2Vec

    #%% Global Variables %%#
        - Texts: PreProcessing text to extract features
        - Language: 'spanish' or 'english'

    #%% LDA Variables %%#
        - topicsNumber: number of topics to choose
        - passesNumber: numer of passes through the corpus during training

    #%% Word2Vec Variables %%#
        - size: dimensionality of the word vectors.
        - window: maximum distance between the current and predicted word within a sentence.
        - min_count: ignores all words with total frequency lower than this.
"""



#%% Import Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import corpora
from gensim.corpora.wikicorpus import IGNORED_NAMESPACES, WikiCorpus, filter_wiki, find_interlinks, get_namespace, utils
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus, MmCorpus
import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
import numpy as np

#%% Statistical Features: Bag of Words
def Bag_of_WordsFT(texts):
    #instantiate the vectorizer
    vectorizer= CountVectorizer(texts)  #to tokenize the document
    #fit them as bag of words
    bag_of_words=vectorizer.fit(texts)
    #apply transofrm method
    bag_of_words=vectorizer.transform(texts).toarray()

    return bag_of_words

#%% Statistical Features: TF-IDF
def TF_IDF(texts):
    obj=TfidfVectorizer()
    tfi_idf=obj.fit_transform(texts).toarray()
    return tfi_idf

#%% Entity Extraction: Topic Modeling-LDA
def LDADictionary(Language='spanish'):
    path_to_wiki_dump = datapath('D:/Gita/GITA_Master/Databases/WikiCorpus/eswiki-latest-pages-articles.xml.bz2')
    corpus_path = get_tmpfile("wiki-corpus.mm")

    wiki = WikiCorpus(path_to_wiki_dump) # create word->word_id mapping, ~8h on full wiki
    MmCorpus.serialize(corpus_path, wiki) # another 8h, creates a file in MatrixMarket format and mapping
    dictionary=wiki.dictionary

    return dictionary

def LDA(texts, dictionary,topicsNumber=100,passesNumber=50, Language='spanish'):

    #Converting list of documents (corpus) into document term Matric using dictionary prepare above.
    doc_term_matrix=[]
    for i in range (len(texts)):
        doc_term_matrix.append(dictionary.doc2bow
        (gensim.corpora.wikicorpus.tokenize(texts[i])))
    #Creating the object for LDA model using gensim library
    Lda=gensim.models.ldamodel.LdaModel
    #Results
    ldamodel= Lda(doc_term_matrix, num_topics=topicsNumber, id2word=dictionary, passes=passesNumber)
    ldaTopics=ldamodel[doc_term_matrix]
    #Extract the value of each topic
    for doc in ldaTopics:
        docs=doc[1]

    return docs

#%% Word Emmbbedings: Word2Vec
def Word2VecTraining(Size=200, window=7, min_count=10, Language='spanish'):

    #Using WikiCorpus in Spanish Version
    wiki = WikiCorpus('D:/Gita/GITA_Master/Databases/WikiCorpus/eswiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary={})
    corpus = list(wiki.get_texts())
    #Defining Paramters
    params = {'size': Size, 'window': window, 'min_count': min_count,}
    #Model Training with WikiCorpus
    word2vec = Word2Vec(corpus, **params)



    return word2vec


def Word2VecFTE(texts, Word2Vec):

    #To save features (each word)
    doc=[]

    for tx in range (len(texts)):
        words=[]
        for w in range (len(texts[tx])):
            words.append(Word2Vec[texts[tx][w]])
        doc.append(np.mean(words))

    return doc
