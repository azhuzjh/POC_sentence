#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from flask import Flask
from flask import Markup
from flask import Flask, render_template,request, url_for
from flask_bootstrap import Bootstrap
import random
import time

# backend needed -- starting
import pandas as pd
import re
#import nltk
#from nltk import word_tokenize
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

import app_util 



# In[20]:


FOLDER_DEST = "../model/saved"
MODELNAME = "MODEL_W2V_CUST"
TRAIN_DATA = "data_live_chat_2019"

NUM_CLUSTER = 20
MODEL_PATH = path.join(FOLDER_DEST, "%s_%s" % (TRAIN_DATA, MODELNAME))
#print(MODEL_PATH)

DATA_FOLDER_DEST = "../data"
SEARCH_DATA_NAME = "data_chat_msg_2019_full_sentences" #temp_app
SEARCH_DATA_PATH = path.join(DATA_FOLDER_DEST, "%s.csv" % SEARCH_DATA_NAME)
#print("\ntest data is following")
#print(SEARCH_DATA_PATH)

MODEL = word2vec.Word2Vec.load(MODEL_PATH)

# backend needed -- ending


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentanalyse', methods=['POST'])
def sentanalyse():

    if request.method =='POST':
        rawtext = request.form['rawtext']
                
        # search data, calling model -- starting
        df = pd.read_csv(SEARCH_DATA_PATH)
        #print(df.shape)
        
        df = df[df.source == "visitor"]
        df = df[df.question_or_not == "question"]
        
        df = df.drop_duplicates(subset = "text",keep='last')
        df.reset_index()
        #print(df.shape)
        
        sim_text = list()
        print(rawtext)

        for t in df.text: 
            sen_vec = app_util.get_sen_vec(t, MODEL)
            #print(sen_vec)
            sim = 1 - spatial.distance.cosine(app_util.get_sen_vec(rawtext, MODEL),sen_vec)
            if sim > 0.9:
                sim_text.append(t)
                print("**", t)
                print(sim)
                
                if len(sim_text) > 20: 
                    break
            
        #print(sim_text)
        
        summary = sim_text

        # search data, calling model -- ending
        
     
        return render_template('index.html', received_text = rawtext, summary=summary)
    
    

if __name__ == '__main__':
	app.debug = True
	app.run()


