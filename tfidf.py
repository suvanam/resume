import re
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import pandas as pd
import numpy as np
import math
import string
import warnings
warnings.filterwarnings("ignore")

#sentence = ("Dogs are very? friendly, animals. All animals@ **    **  are not dog ?? 009. Cats can be closure to dogs and monkeys.")

def tf_idf(sentence):
    
    sentence=sentence.replace(" ,", ", ")
    sentence=sentence.replace(" .", ". ")
    sentence=sentence.replace(" . ", ". ")
    sentence=sentence.replace(" , ", ", ")
    
    sentence = re.sub('\\s+', ' ', sentence)    # remove extra blank spaces 
    
    # remove stopwords
    stop_words=nltk.corpus.stopwords.words('english')
    AddNewStopWords = ['etc', 'etc.',
                       'kms', 'km', 'hrs', 'hours', 'mld', 'litres', 'kilo', 'meters', 'cm',
                       'mm', 'usd', 'dollar', "dollars", "INR", 'crore', 'crores', 'million', 'billion',
                       'millions', 'billions','lakhs', 'lacs', 'lakh', 'lac', 'cum', 'kg', 'kgs',
                       'gram', 'grams', 'currency', 'length', 'width', 'height',
                       'volume', 'area','ft', 'feet', 'sqft', 'sqm', 'sq.ft', 'sq.m']
    stop_words.extend(AddNewStopWords)  
    
    sent_stopword= " ".join([word for word in sentence.split() if word.lower() not in stop_words]) 
    sent_stopword=sent_stopword.lower()
    
    # Lemmatise and create vocabulary 
    s_for_count=re.sub("[^a-zA-Z]", " ", sent_stopword)# clear punctuation. 
    s_for_count=re.sub(r'[0-9]', " ", s_for_count)# clear numbers
    
    nlp = spacy.load('en_core_web_sm')
    doc=nlp(s_for_count)
    sent_lemma=' '.join([token.lemma_ for token in doc]) 
    bow_tf=sent_lemma  # to be used for key words in bow at end of code

    word_lemma=nltk.word_tokenize(sent_lemma)
    word_df= pd.DataFrame(word_lemma, columns = ['lemma_word']) # create all words lemma
    word_df=word_df.drop_duplicates(subset=["lemma_word"], keep='first') # remove duplicate words keep first
    
    
    # create lemma word-sentence one hot values
    
    x=nltk.sent_tokenize(sentence)
    i=0

    for sent in x:
        doc=nlp(sent)
        y=' '.join([token.lemma_ for token in doc]).lower()
        sent= re.sub("[^a-zA-Z]", " ", y)# clear sentence period (stop) for that sentence
        w=nltk.word_tokenize(sent)
        word_df[str(i)]=0  # create new column for sentence index i
        for ele in w:   
            word_df.loc[(word_df['lemma_word']== ele), str(i)] = 1
        i=i+1
    #print(word_df)
    num_sentences=len(word_df.columns)-1  # use to compare number of sentenc columns with number of sentence for error
    
    
    # calculate IDF of words
    lem_list=word_df['lemma_word'].tolist()
    x=nltk.word_tokenize(sent_lemma)
    total_doc_words=len(x) # total document words after lemmatisation
    
    idf_values=[]
    for ele in lem_list: 
        freq=x.count(ele) 

        if freq>0:   # a number string is removed in lemmaization so it will appear as sentence. This if used to eliminate zero division
            idf= math.log2((total_doc_words)/(freq))
        else:
            idf=0  # freq zero means word not exisiting so no idf
            
        y=[ele, round(idf,2)]
        idf_values.append(y)
    df_idf= pd.DataFrame (idf_values, columns = (['lemma_word','idf_value'])) # create idf table
    
    if len(word_df)==len(df_idf):
        word_idf= pd.merge(df_idf, word_df, on='lemma_word', how='inner')
        
    else:
        print("Error encountered while reformatting document. Cannot be done...")
        text=""
        return(text)

    
    # calculate IDF of sentences
    word_idf_master=word_idf
    x=nltk.sent_tokenize(sentence)
    
    n=len(x)
    #print("n",n)
    sentIDF_val=[]

    if n!=num_sentences:
        print("Error encountered while reformatting document. Cannot be done...")
        text=""
        return(text)

    for i in range(n):
        
        num_of_words= sum(word_idf[str(i)]) # number of words. 
        word_idf[str(i)]=word_idf['idf_value']* word_idf[str(i)] # This is should be done after num_of_words
        idf_sum=round(sum(word_idf[str(i)]),2) # idf value of sent by sum of words in sent rounded to 2 deci

        if num_of_words>0:  # a number string is removed in lemmaization so it will appear as sentence. This if used to eliminate zero division
            idf_word=round(idf_sum/num_of_words,2)  # calculate idf per unique lemma word. rounded to 2 dec
        else:
            idf_word=0
            
        y=[i, x[i],idf_sum, num_of_words, idf_word]
        
        sentIDF_val.append(y) # this is the IDF values for all sentences, total words and idf per word

    sentence_idf_df= pd.DataFrame (sentIDF_val, columns = (['id','sentence','idf_val', "words", "idf_per_word"]))
    sentence_tf=sentence_idf_df['sentence'].tolist() # carry to main module
    
    # Filter out short and long sentences, and high and low  idf per word to eliminate outliers

    
    mx=sentence_idf_df['idf_per_word'].max()
    mn=sentence_idf_df['idf_per_word'].min()
    mid=(mx+mn)/2
    cutoffmx=.05
    cutoffmn=.3
    cutoff_max=mx-(mx-mid)*cutoffmx
    cutoff_min=mn+(mid-mn)*cutoffmn
    df_filter=sentence_idf_df
    
    df_filter=df_filter.loc[df_filter['idf_per_word'] < cutoff_max+.0001]
    df_filter=df_filter.loc[df_filter['idf_per_word'] > cutoff_min-.0001]
    

    #print(mx,mn,mid,cutoff_max,cutoff_min)
    dftemp=df_filter['idf_per_word'].tolist()
    dftemp=df_filter['words'].tolist()
    
    
    df_tfidf=df_filter.sort_values(by=['id'])
    
    
    # send df to list
    text=df_tfidf['sentence'].tolist()
    text= " ".join(map(str, text))
    text=text.replace(" .", ". ")

    text=text.replace(" .", ". ")
    text=text.replace(" ,", ", ")
    text=text.replace(" . ", ". ")
    text=text.replace(" , ", ", ")

    # create bag of words with count for key words in main
    text_b=bow_tf.lower()
    text_w=nltk.word_tokenize(text_b)
    bow=[]
    for i in range(len(text_w)-1):
        if (len(text_w[i])>2 and len(text_w[i+1])>2):
            s=text_w[i]+" "+text_w[i+1]
            n=text_b.count(s)
            k=[s,n]
            bow.append(k)
    df_bow= pd.DataFrame(bow, columns = ['pair', 'count'])
    df_bow=df_bow.drop_duplicates(subset=["pair"], keep='first')
    df_sort=df_bow.sort_values(by=['count'])
    df_bow=df_sort.tail(12)  # take 25 for Yake
    bow=df_bow['pair'].tolist()
    bow=", ".join(map(str,bow))
    bow=bow.title()

    return text, bow,sentence_tf
#r=tf_idf(sentence)
