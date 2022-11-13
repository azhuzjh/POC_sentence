#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import re
import nltk
from nltk import word_tokenize
import string
from string import digits
import pickle
import gensim
from gensim.models import word2vec
from gensim.models import Phrases
import scipy
import sklearn
import operator
from collections import Counter
import json
import numpy as np
from pandas import DataFrame
import warnings
warnings.filterwarnings("ignore")
from os import path
from scipy import spatial

STOP_WORDS = pickle.load(open("../data/stopwords.pkl", 'rb'))
#print(len(STOP_WORDS))


# In[20]:


FOLDER_DEST = "../model/saved"
MODELNAME = "MODEL_W2V_CUST"
TRAIN_DATA = "data_live_chat_2019"

NUM_CLUSTER = 20
MODEL_PATH = path.join(FOLDER_DEST, "%s_%s" % (TRAIN_DATA, MODELNAME))
print(MODEL_PATH)

DATA_FOLDER_DEST = "../data"
#SEARCH_DATA_NAME = "data_chat_msg_2019_Q"
SEARCH_DATA_NAME = "data_chat_msg_2019_full_sentences"
SEARCH_DATA_PATH = path.join(DATA_FOLDER_DEST, "%s.csv" % SEARCH_DATA_NAME)
print("\ntest data is following")
print(SEARCH_DATA_PATH)

MODEL = word2vec.Word2Vec.load(MODEL_PATH)

def build_feature (sentences):
    sentences = [re.sub("[^a-zA-Z]", " ", sent) for sent in sentences]
    sentences = [[i for i in word_tokenize(sent) if i not in string.punctuation and i not in digits and len(i) > 2] for sent in sentences]
    sentences = [[i.lower() for i in x] for x in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    sentences = [[item[0] for item in tag if item[1][0] == 'N' or item[1][0] == 'V' or item[1][0] == 'W' or item[1][0] == 'J' and item[0] not in STOP_WORDS] for tag in sentences]
    sentences = [i for i in sentences if len(i) > 2 ]
    return sentences

def avg_vector(words, model, num_features = 1000):
    '''
    output: an vector averaged from all word vectors
    '''
    avg_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    for word in words:
        if word in set(model.wv.vocab):
            nwords = nwords+1
            avg_vec = np.add(avg_vec, model[word])
    if nwords > 0:
        avg_vec = np.divide(avg_vec, nwords)
    return avg_vec


def get_sen_vec(text, model): 
    features = build_feature([text])[0]
    sen_vec = avg_vector(features, model)
    return sen_vec


# In[29]:


TEST = 'What concealer works best with this foundation?'
#TEST = 'what is the difference between new and old version of this foundation?'
#TEST = 'which of the foundation matches closely with my dark skin tone'


# In[30]:


df = pd.read_csv(SEARCH_DATA_PATH)
df.head(5)
print(df.shape)
df = df[df.source == "visitor"]
df = df[df.question_or_not == "question"]
print(df.shape)


# In[31]:


sim_text = []
sim_score = []

for t in df.text: 
    try: 
        sen_vec = get_sen_vec(t, MODEL)
        sim = 1 - spatial.distance.cosine(get_sen_vec(TEST, MODEL),sen_vec)
        if sim > 0.9:
            sim_text.append(t)
            sim_score.append(sim)
        if len(sim_text) > 10: 
            break
    except: 
        pass
    
print(sim_text)



# In[ ]:




