print('Importing libraries...')
import os
import clean
import nltk
import spacy
from pdfminer.high_level import extract_text
import pytextrank
import re
import tfidf
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
import string
import numpy as np
import yake
from yake import KeywordExtractor
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def main_start(filename):
    np.random.seed(1)

    #***********FILE SELECT, CLEAN, TFIDF**********************
    

    #  Get pdf, convert to text and clean
    rawtext = extract_text(filename) # read pdf file
    cleantext=clean.clean_raw(rawtext) # output from clean.py
    text_textrank, bow_tf, sentence_tf=tfidf.tf_idf(cleantext)  # bow_tf from tfidf for key phrases


    #************RUN TEXTRANK*********************************
    sent_phr=500 #No of phrases
    sent_num= 10 # No of sentences

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    mystring=[]
    t=nlp(cleantext)
    for sent in t._.textrank.summary(limit_phrases=sent_phr,limit_sentences=sent_num): # limit_phrases=sent_phr,
        mystring.append(sent)

    mystring=set(mystring)
    summary=" ".join(map(str,mystring))

    #*************GET BAG OF WORDS FROM TFIFD for YAKE***********************************
    language = "en"
    max_ngram_size = 2
    deduplication_thresold = .9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords =10

    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, stopwords=None)
    keywords = kw_extractor.extract_keywords(bow_tf) 

    t=[]
    for kw, v in keywords:  # select 2 or more phrases in keyword. Limit to 7
        t.append(kw)
    keywords=", ".join(map(str,t))
    keywords=keywords.title()


    #************CLEAN OUTPUTS************************************************
    summary=summary.replace(",.", ". ")
    summary=summary.replace(". .", ". ")
    summary=summary.replace(" .", ". ")
    summary=summary.replace(" . ", ". ")
    summary=summary.replace("..", ". ")
    summary=summary.replace(" , ", ", ")
    summary=summary.replace(" ,", ", ")
    summary = re.sub('\\s+', ' ', summary) # remove extra blank spaces


    #**************PRINT OUTPUTS*************************************************************
    #print("\n")
    #print("*"*20,"SUMMARY","*"*20)
    #print("\n")
    #print(summary)
    #print("\n")
    #print("*"*20,"KEY PHRASES","*"*20)
    #print("\n")
    #print(keywords)
    #print("\n")
    return summary, keywords

global folder_path
filename = st.text_input('Enter a file path:')
s=[]
k=[]
f=[]
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".pdf"):
        file_path = f"{path}\{file}"
        print("Processing resume: ", file_path)
        summ, key=main_start(file_path)
        s.append(summ)
        k.append(key)
        f.append(file_path)
q=[]
for i in range(len(f)):
    w=[f[i],s[i],k[i]]
    q.append(w)
df=pd.DataFrame (q, columns = (['fileno','summary', 'key']))
df = pd.read_excel()
df.to_excel('resume_summary.xlsx')
print("Saved into Excel as file: resume_summary.xlsx")

