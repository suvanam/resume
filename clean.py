import nltk
import re
import string
#import locationtagger
import nltk
from nltk import sent_tokenize, word_tokenize
import pandas as pd
import unicodedata
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def clean_raw(rawtext):
    print("Processing document...")  
#   remove \n and reconstruct sentences
#   if next sentence starts with upper alphabet then previous sentence ends.
#   if not next sentence continuation of previous. Not perfect logic though.

    
    rawtext=re.sub('[^a-zA-Z0-9 \n\,.]','',rawtext)
    
    
    def is_pua(c): # remove \uf02d etc
        return unicodedata.category(c) == 'Co'
    t="".join([char for char in rawtext if not is_pua(char)]) 

    #remove escape sequence
    t = t.replace(r'[\x00-\x1f]+', '')
    
    punc = '''____!()-[]{};:'"\<>/?@#$%^&*_~'''
    t="".join([c for c in t if c not in punc])

    t=re.sub('\n \n', '\n', t)
    t=re.sub('\n\n', '\n', t)
    t=re.sub('\n\n\n', '\n', t)
    t=re.sub('\n\n\n\n', '\n', t)
    text = re.sub('\\s+', ' ', rawtext)    # remove extra blank spaces
    text=re.sub("(cid:[0-9]+)", '', text) # Remove non-unicode
    
    
    t=t.splitlines()

    # strip white space elements
    
   
    
    df=pd.DataFrame (t, columns = (['sentence']))
    
    df['sent']=df['sentence'].str.strip()
    
    df=df.loc[df['sent'] !=""]
    df=df.loc[df['sent'] !=" "]
    df=df.loc[df['sent'] !="  "]
    df=df.loc[df['sent'] !="   "]
    

    x=df['sent'].tolist() # To avoid key error as some rows deleted in df as above
    
    df=pd.DataFrame (x, columns = (['sent']))

   
    x=[]
    k=0
    for i in range(len(df)): # this block checks if next sentence starts with uppercase, it is a new sentencee
        if i<len(df)-1:   
            a=df.loc[i+1,'sent']
            
            if a.split()[0][0].isupper():
                k=1
            if a.split()[0][0].islower():
                k=0

        if i==len(df)-1:
            k=1
        x.append(k)    

    df['upper']=x

    x=[]
    k=0
    for i in range(len(df)): # this block checks if a sentence ends with a comma
        if i<len(df)-1:   
            a=df.loc[i,'sent']
            p=len(a.split())
            q=len(a.split()[p-1])
            if a.split()[p-1][q-1]==',':
                k=1
            if a.split()[p-1][q-1]!=',':
                k=0

        if i==len(df)-1:
            k=0
        x.append(k)
    df['comma']=x
    
    df['sent']=np.where(((df['upper'] ==1) & (df['comma'] ==0)), df['sent']+". ", df['sent'])
    text=df['sent'].tolist()
    text=" ".join(map(str,text))
#   Remove emails and websites
    text = re.sub(r'https?:\/\/.\S+', " ", text)
    text = re.sub(r'\S*@\S*\s*', ' ', text)

# Remove duplicate sentences
    text=nltk.sent_tokenize(text)
    df_cleantext= pd.DataFrame (text, columns = (['sentence'])) # create clean text table
    df_cleantext=df_cleantext.drop_duplicates(subset=["sentence"], keep='first') # remove duplicate sentences

# filter out short and long sentences
    max_words=25
    min_words=0
    df_cleantext['words']=df_cleantext['sentence'].str.split().str.len()
    df_cleantext=df_cleantext.loc[df_cleantext['words'] <max_words]
    df_cleantext=df_cleantext.loc[df_cleantext['words'] >min_words]    
    
    text=df_cleantext['sentence'].tolist()
    
# format
    text=" ".join(map(str,text))
    
    text=text.replace("..", ". ")
    text=text.replace(",.", ". ")
    text=text.replace(".,", ", ")
    return text


#w=clean_raw(temp)
