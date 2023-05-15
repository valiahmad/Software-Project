from _.Color import *
print(BOLD+fgray+bwhite+' Preprocessing'+End)

import string
import torch
import pandas as pd
import numpy as np
import json
from _.settings import setting
from _.datasetParser import Parse
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from spellchecker import SpellChecker
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn

# Reading the settings
Settings = setting()
path = 'integrated_dataset.xlsx'


def Preprocess():

    # Loading Data
    if Settings['Excel File']:
        df = pd.read_excel(path)
        print(df.head())
    else:
        df = Parse()
        df = df.returnType(dataFrameMode=True)
        print(df.head())

    

    # To Lowercase
    if Settings['Lowercase']:
        df.loc[df['Category'] == 'Hu&Liu', 'Review'] = df['Review'].str.lower()
        
    # Punctuations Removal !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    if Settings['Punctuation']:
        df.loc[df['Category'] == 'Hu&Liu', 'Review'] = df['Review']\
        .str.translate(str.maketrans('', '', string.punctuation))
        print(SIMP+fblue+bgray+'\n Punctuation Removal Done!'+End)

    # Digit Removal
    if Settings['Digit']:
        df['Review'] = df['Review'].str.translate(str.maketrans('', '', string.digits))
        print(SIMP+fblue+bgray+'\n Digit Removal Done!'+End)

    # Reviews Removal of less than 10 letters
    if Settings['<10letters']:
        df = df.drop(df[df['Review'].str.len() < 10 and df['Category'] == 'Hu&Liu'].index)
        print(SIMP+fblue+bgray+'\n Less Than 10 Letters Reviews Droped!'+End)

    # Tokenized reviews text
    if Settings['Tokenization']:
        df.loc[df['Categoty'] == 'Hu&Liu', 'Tokenized'] = df['Review'].apply(word_tokenize)
        print(SIMP+fblue+bgray+'\n Tokenization Done!'+End)
    else:
        print(BOLD+fred+bgray+'\nData is NOT Tokenized!!!'+End)
        input('\nTo Continue Press Enter...')

    # BERT Tokenization
    if Settings['BERT-Tokenization']:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # padding='max_length',max_length=64,return_tensors="pt",return_attention_mask=True
        df['BERT-Tokenized'] = df['Review'].apply(tokenizer.tokenize)
        print(BOLD+fblue+bgray+'\n BERT-Tokenization Done!'+End)
    else:
        print(BOLD+fred+bgray+'\nData is NOT Tokenized(BERT)!!!'+End)
        input('\nTo Continue Press Enter...')
    # TODO : for BERT_
    # Correcting word spells
    if Settings['Spell Checking']:
        spell = SpellChecker()
        df['Tokenized'] = df['Tokenized'].apply(lambda x: [spell.correction(word) for word in x])
        df['Tokenized'] = df['Tokenized'].apply(lambda x: [word for word in x if word is not None])
        print(SIMP+fblue+bgray+'\n Spell Checked!'+End)
    else:
        print(BOLD+fred+bgray+'\nData is NOT Spell-Checked!!!'+End)
        input('\nTo Continue Press Enter...')

    # Tagging Part-of-Speech
    if Settings['POS Tagging']:
        df['Tagged'] = df['Tokenized'].apply(pos_tag)
        print(SIMP+fblue+bgray+'\n POS Tagged!'+End)

    # StopWords Removing
    if Settings['StopWords']:
        stop_words = set(stopwords.words('english'))
        df['StopWordsRemoved'] = df['Tagged'].apply(
            lambda x: [word for word in x if word[0].lower() not in stop_words])
        print(SIMP+fblue+bgray+'\n StopWords Removal Done!'+End)

    # Stemming
    if Settings['Stemming']:
        stemmer = SnowballStemmer("english")
        df['Stemmed'] = df['StopWordsRemoved'].apply(
            lambda x: [(stemmer.stem(word[0]),word[1]) for word in x])
        print(SIMP+fblue+bgray+'\n Words Stemmed!'+End)

    # Lemmatization
    if Settings['Lemmatization']:
        wnl = WordNetLemmatizer()
        df['Lemmatized'] = df['Stemmed'].apply(lambda x: [wnl.lemmatize(word[0], word[1]) for word in x])
        print(SIMP+fblue+bgray+'\n Lemmatization Done!'+End)

    # Fixing the problems with dataset
    if Settings['Fix Problem']:

        print(SIMP+fblue+bgray+'\n Problems With The Dataset Fixed!'+End)
    
    # Formatting for BERT
    if Settings['BERT Format']:
        df['BERT-Tokenized'] = df['BERT-Tokenized'].apply(lambda x: ['[CLS]'] + x + ['[SEP]'])
        df['TokenID'] = df['BERT-Tokenized'].apply(tokenizer.convert_tokens_to_ids)
        df['SegmentID'] = df['BERT-Tokenized'].apply(lambda x: [1]*len(x))
        df['Token-Tensor'] = df['TokenID'].apply(lambda x: torch.tensor([x]))
        df['Segment-Tensor'] = df['SegmentID'].apply(lambda x: torch.tensor([x]))
        print(SIMP+fblue+bgray+'\n Data Prepared For BERT!'+End)
    
    

    return df







def Concat(df: pd.DataFrame, col: str, i: int):
    if (len(df)-1) == i:
        return df[col].iloc[i]
    return np.concatenate((df[col].iloc[i], Concat(df, col, i+1)), axis=0)







def Split(arr: np.array, df: pd.DataFrame, col: str):
    l = []
    pos = 0
    for i in range(len(df)):
        l.append(list(arr[pos:len(df[col].iloc[i])]))
        pos += len(df[col].iloc[i])
    
    return l







def getID(n):
    
    while(True):
        arr = np.random.randint(10000, 999999, n*10)
        if len(set(arr)) >= n:
            return np.array(list(set(arr)))







def setID(df: pd.DataFrame, col: str):
    n = 0
    for i in range(len(df)):
        n += len(df[col].iloc[i])

    IDs = getID(n)

    return Split(IDs, df, col)
    






def freqDist(df: pd.DataFrame, cols: list):
    # {Label:{Word:FreqDist}}
    fd = {}
    Vectors_Clusters = cols[0]
    Tagged_Sentences = cols[1]

    for i in range(len(df)):
        for j in range(len(df[Vectors_Clusters].iloc[i])):
            if df[Vectors_Clusters].iloc[i][j] in fd:
                if df[Tagged_Sentences].iloc[i][j][0] in fd[df[Vectors_Clusters].iloc[i][j]]:
                    fd[df[Vectors_Clusters].iloc[i][j]][df[Tagged_Sentences].iloc[i][j][0]] += 1
                else:
                    fd[df[Vectors_Clusters].iloc[i][j]][df[Tagged_Sentences].iloc[i][j][0]] = 1
            else:
                fd[df[Vectors_Clusters].iloc[i][j]] = {df[Tagged_Sentences].iloc[i][j][0] : 1}

    return fd







def distSpace(df: pd.DataFrame, cols: list):
    # {Label:[[id, coordinate, Dist-Space]...]}
    ds = {}
    Vectors_Clusters = cols[1]
    IDs = cols[0]
    Coordinates = cols[2]

    for i in range(len(df)):
        for j in range(len(df[IDs].iloc[i])):
            if df[Vectors_Clusters].iloc[i][j] in ds:
                ds[df[Vectors_Clusters].iloc[i][j]].append([
                    df[IDs].iloc[i][j],
                    df[Coordinates].iloc[i][j]
                ])
            else:
                ds[df[Vectors_Clusters].iloc[i][j]] = [df[IDs].iloc[i][j], df[Coordinates].iloc[i][j]]

    with open('centers_clusters.json') as file:
        cc = json.load(file)
    cc = cc[Vectors_Clusters]

    # CC[0]->Label 0
    cor = 1 #Coordinate index
    for label in range(len(cc)):
        for feature in range(len(ds[label])):
            ds[label][feature].append(
                np.sqrt(
                (cc[label][0] - ds[label][feature][cor][0])**2 
                + (cc[label][1] - ds[label][feature][cor][1])**2)
            )

    return ds







def prepItems(df: pd.DataFrame, cols: list):
    # {id:word}
    d = {}
    Tagged_Sentences = cols[0] 
    IDs = cols[1]
    
    for i in range(len(df)):
        for j in range(len(df[IDs].iloc[i])):
            d[df[IDs].iloc[i][j]] = df[Tagged_Sentences].iloc[i][j][0]

    return d







def Polarity(doc: list):

    pos = 0
    neg = 0
    rate = .013
    count = 0
    
    for word in doc:
        sentiSet = list(swn.senti_synsets(word))
        if len(sentiSet) == 0:
            continue
        else:
            count += 1
            sentiSet0 = sentiSet[0]
            pos += sentiSet0.pos_score()
            neg += sentiSet0.neg_score()
    
    if count:
        distance = abs((pos/count) - (neg/count))
    else:
        distance = 0
    
    if distance < rate:
        return 0
    elif pos > neg:
        return 1
    else:
        return -1