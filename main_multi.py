print('Importing libraries...')
import os
import clean
import nltk
import spacy
from spacy.matcher import Matcher
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
from docx2pdf import convert
import shutil
import warnings
warnings.filterwarnings("ignore")
import time


def main_start(filename):
    
    np.random.seed(1)
    
    #***********FILE SELECT, CLEAN, TFIDF**********************
    

    #  Get pdf, convert to text and clean
    rawtext = extract_text(filename) # read pdf file
    cleantext,name_text=clean.clean_raw(rawtext) # output from clean.py
    text_textrank,keytext=tfidf.tf_idf(cleantext)  # bow_tf from tfidf for key phrases
    

    #************RUN TEXTRANK*********************************
    sent_phr=500 #No of phrases
    sent_num= sentnum # No of sentences

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    mystring=[]
    t=nlp(text_textrank)
    for sent in t._.textrank.summary(limit_phrases=sent_phr,limit_sentences=sent_num): # limit_phrases=sent_phr,
        mystring.append(sent)

    mystring=set(mystring)
    summary=" ".join(map(str,mystring))

    #*************YAKE ON SUMMARY**********************************

    language = "en"
    max_ngram_size = 2
    deduplication_thresold = .99
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords =12

    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, stopwords=None)
    keywords = kw_extractor.extract_keywords(keytext) 
    
    t=[]
    for kw, v in keywords:
        k=kw.split()
        if len(k)==2:
            if k[0]!=k[1]:
                t.append(kw)
        
    keywords=", ".join(map(str,t))
    keywords=keywords.title()
    print(keywords)

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
    return summary, rawtext. keywords

def name_extract(rawtext):
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    pattern = [[{"POS": "PROPN"}]+[{"POS": "PROPN"}]]
    matcher.add("My_Pattern",pattern)

    text=rawtext.splitlines()
    text = text[:1]
    text=" ".join(map(str,text))
    doc = nlp(text)
    matches = matcher(doc)

    #for match_id,start,end in matches:
        #print(doc[start:end]for match_id,start,end in matches)
    return ([doc[start:end]for match_id,start,end in matches])


path = input('Enter folder path containing resumes:')
sentnum=int(input("How many sentences in summary that you want (min 10, max 20)?"))
# convert all docx to pdf
convert(f"{path}")

print('\n')
s=[]
k=[]
f=[]
nm=[]
em=[]

for file in os.listdir(f"{path}"):
    # Check whether file is in text format or not

        
    if file.endswith(".pdf"):
        file_path = f"{path}/{file}"
        print("Processing resume: ", file_path)
        time_start=time.time()
        summ, rawtext, keywords=main_start(file_path)
        d= name_extract(rawtext)
        # email
        r = re.compile(r'[\w\.-]+@[\w\.-]+')
        email=r.findall(rawtext)
        email=str(email)

        if d!=[]:
            print(d)
        if d==[]:
            print("Name not traced")

        if email!="":
            print(email)

        if email=="":
            print("Email not traced")
            
        s.append(summ)
        k.append(keywords)
        f.append(file_path)
        nm.append(d)
        em.append(email)
        
        print("Timetaken:", int(time.time()-time_start)," seconds")
        
        print('\n')
q=[]
for i in range(len(f)):
    w=[f[i],nm[i],em[i],s[i],k[i]]
    q.append(w)
df=pd.DataFrame (q, columns = (['file','name','email','summary', 'key']))

# check if resume_summary.xlsx exists, then remove

if  os.path.isfile(path+'/resume_summary.xlsx'):
        os.remove(path+'/resume_summary.xlsx')
        

df.to_excel("resume_summary.xlsx", sheet_name='summary')

# move to resumes folder
directory_path = os.getcwd()
src_path = os.path.abspath(directory_path)+'\\resume_summary.xlsx'
dst_path = path+'\\resume_summary.xlsx'
shutil.move(src_path, dst_path)

print("Saved in folder ",f"{path}", "File name: resume_summary.xlsx")
print
