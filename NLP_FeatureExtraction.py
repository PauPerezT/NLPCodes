"""
@Author: Paula Andrea Pérez Toro
@Date:   2018-11-28T11:41:08-05:00
@Email:  paulaperezt16@gmail.com
@Filename: NLP_FeatureExtraction.py
# @Last modified by:   Paula Andrea Pérez Toro
# @Last modified time: 2018-11-28T17:25:48-05:00

"""

#%% Import Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import corpora
from gensim.corpora.wikicorpus import IGNORED_NAMESPACES, WikiCorpus, filter_wiki, find_interlinks, get_namespace, utils
import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec


#%% Statistical Features: Bag of Words
def Bag_of_WordsFT(textx):
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
    tfi_idf=obj.fit_transform(corpus).toarray()
    return tfi_idf

#%% Entity Extraction: Topic Modeling-LDA
def LDA(texts,topicsNumber=100,passesNumber=50):
    path_to_wiki_dump = datapath('D:/Gita/GITA_Master/Databases/WikiCorpus/eswiki-latest-pages-articles.xml.bz2')
    corpus_path = get_tmpfile("wiki-corpus.mm")

    wiki = WikiCorpus(path_to_wiki_dump) # create word->word_id mapping, ~8h on full wiki
    MmCorpus.serialize(corpus_path, wiki) # another 8h, creates a file in MatrixMarket format and mapping
    #Converting list of documents (corpus) into document term Matric using dictionary prepare above.
    doc_term_matrix=[]
    for i in range (len(texts)):
        doc_term_matrix.append(wiki.dictionary.doc2bow
        (gensim.corpora.wikicorpus.tokenize(texts[i])))
    #Creating the object for LDA model using gensim library
    Lda=gensim.models.ldamodel.LdaModel
    #Results
    ldamodel= Lda(doc_term_matrix, num_topics=topicsNumber, id2word=wiki.dictionary, passes=passesNumber)
    ldaTopics=ldamodel[doc_term_matrix]
    #Extract the value of each topic
    for doc in ldaTopics:
        docs=doc[1]

    return docs

#%% Word Emmbbedings: Word2Vec
def Word2Vec(texts, Size=200, window=10, min_count=10):

    #Using WikiCorpus in Spanish Version
    wiki = WikiCorpus('D:/Gita/GITA_Master/Databases/WikiCorpus/eswiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary={})
    corpus = list(wiki.get_texts())
    #Defining Paramters
    params = {'size': Size, 'window': window, 'min_count': min_count,
              'workers': max(1, multiprocessing.cpu_count() - 1),}
    #Model Training with WikiCorpus
    word2vec = Word2Vec(corpus, **params)

    #To save features (each word)
    doc=[]

for tx in range (len(texts)):
    words=[]
    for w in range (len(texts[tx])):
        words.append(model[texts[tx][w]])
    doc.append(words)

    ###I HAVE TO FINISH IT####
