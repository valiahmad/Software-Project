from _.Color import *
print(BOLD+fgray+bwhite+' Preprocessing'+End)
import string
import torch
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from spellchecker import SpellChecker
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from parameters import n_sample




def Preprocess(item: dict):

    # Loading Data
    path_Laptops = './Datasets/Laptops.xlsx'
    path_Rest = './Datasets/Restaurant.xlsx'
    dfL = pd.read_excel(path_Laptops)
    dfR = pd.read_excel(path_Rest)
    dfL['Category'] = 'Laptops'
    dfR['Category'] = 'Restaurant'
    df = pd.concat([dfL, dfR], ignore_index=True)
    df = df.rename(columns=
                    {'id':'SenID',
                    'Sentence':'Review',
                    'Aspect Term':'Feature',
                    'polarity':'Polarity'}
                )
    df = df.drop(columns=['from', 'to'])
    if n_sample:
        df = df.sample(n=n_sample)
    print(df.head())
    
  

    for i in range(1, len(item)+1):
        if i in item:

            # To Lowercase
            if item[i] == 'Lowercase':
                df['Review'] = df['Review'].str.lower()
                print(SIMP+fblue+bgray+'\n Lowercase Done!'+End)
                

            # Punctuations Removal !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
            elif item[i] == 'Punctuation':
                df['Review'] = df['Review'].str.\
                translate(str.maketrans('', '', string.punctuation))
                print(SIMP+fblue+bgray+'\n Punctuation Removal Done!'+End)
                

            # Digit Removal
            elif item[i] == 'Digit':
                df['Review'] = df['Review'].str.translate(str.maketrans('', '', string.digits))
                print(SIMP+fblue+bgray+'\n Digit Removal Done!'+End)
                

            # Reviews Removal of less than 10 letters
            elif item[i] == '<10letters':
                df = df.drop(df[df['Review'].str.len() < 10].index)
                print(SIMP+fblue+bgray+'\n Less Than 10 Letters Reviews Dropped!'+End)
                

            # Tokenized reviews text
            elif item[i] == 'Tokenization':
                df['Tokenized'] = df['Review'].apply(word_tokenize)
                print(SIMP+fblue+bgray+'\n Tokenization Done!'+End)
                
            
            # BERT Tokenization
            elif item[i] == 'BERT-Tokenization':
                # Base
                Btokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                # padding='max_length',max_length=64,return_tensors="pt",return_attention_mask=True
                df['BERT-Tokenized-Base'] = df['Review'].apply(Btokenizer.tokenize)
                
                # Large
                Ltokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
                df['BERT-Tokenized-Large'] = df['Review'].apply(Ltokenizer.tokenize)
                print(SIMP+fblue+bgray+'\n BERT-Tokenization Done!'+End)
                

            # Correcting word spells
            elif item[i] == 'Spell Checking':
                spell = SpellChecker()
                df['Tokenized'] = df['Tokenized'].apply(lambda x: [spell.correction(word) for word in x])
                df['Tokenized'] = df['Tokenized'].apply(lambda x: [word for word in x if word is not None])
                print(SIMP+fblue+bgray+'\n Spell Checked!'+End)
                
            
            # Tagging Part-of-Speech
            elif item[i] == 'POS Tagging':
                df['Tokenized'] = df['Tokenized'].apply(pos_tag)
                df['Tokenized'] = df['Tokenized'].apply(
                    lambda x: [w for w in x if 'JJ' in w[1] or 'NN' in w[1] or 'VB' in w[1]])
                df = df[df['Tokenized'].str.len() != 0]

                # BERT-Base
                df['BERT-Tokenized-Base'] = df['BERT-Tokenized-Base'].apply(pos_tag)
                df['BERT-Tokenized-Base'] = df['BERT-Tokenized-Base'].apply(
                    lambda x: [w for w in x if 'JJ' in w[1] or 'NN' in w[1] or 'VB' in w[1]])
                df = df[df['BERT-Tokenized-Base'].str.len() != 0]
                # BERT-Large
                df['BERT-Tokenized-Large'] = df['BERT-Tokenized-Large'].apply(pos_tag)
                df['BERT-Tokenized-Large'] = df['BERT-Tokenized-Large'].apply(
                    lambda x: [w for w in x if 'JJ' in w[1] or 'NN' in w[1] or 'VB' in w[1]])
                df = df[df['BERT-Tokenized-Large'].str.len() != 0]
                print(SIMP+fblue+bgray+'\n POS Tagged!'+End)
                

            # StopWords Removing
            elif item[i] == 'StopWords':
                stop_words = set(stopwords.words('english'))
                df['Tokenized'] = df['Tokenized'].apply(
                    lambda x: [word for word in x if word[0].lower() not in stop_words])
        
                # BERT-Base
                df['BERT-Tokenized-Base'] = df['BERT-Tokenized-Base'].apply(
                    lambda x: [word for word in x if word[0].lower() not in stop_words])
                # BERT-Large
                df['BERT-Tokenized-Large'] = df['BERT-Tokenized-Large'].apply(
                    lambda x: [word for word in x if word[0].lower() not in stop_words])
                print(SIMP+fblue+bgray+'\n StopWords Removal Done!'+End)
                

            # Stemming
            elif item[i] == 'Stemming':
                stemmer = SnowballStemmer("english")
                df['Tokenized'] = df['Tokenized'].apply(
                    lambda x: [(stemmer.stem(word[0]),word[1]) for word in x])
                
                # BERT-Base
                df['BERT-Tokenized-Base'] = df['BERT-Tokenized-Base'].apply(
                    lambda x: [(stemmer.stem(word[0]),word[1]) for word in x])
                # BERT-Large
                df['BERT-Tokenized-Large'] = df['BERT-Tokenized-Large'].apply(
                    lambda x: [(stemmer.stem(word[0]),word[1]) for word in x])
                print(SIMP+fblue+bgray+'\n Words Stemmed!'+End)
                

            # Lemmatization
            elif item[i] == 'Lemmatization':
                wnl = WordNetLemmatizer()
                df['_Tokenized'] = df['Tokenized'].apply(
                    lambda x: [w + tuple('a') for w in x if 'JJ' in w[1]])
                df['_Tokenized'] += df['Tokenized'].apply(
                    lambda x: [w + tuple('v') for w in x if 'VB' in w[1]])
                df['_Tokenized'] += df['Tokenized'].apply(
                    lambda x: [w + tuple('n') for w in x if 'NN' in w[1]])
                df['Tokenized'] = df['_Tokenized']
                df = df.drop(columns=['_Tokenized'])
                df['Tokenized'] = df['Tokenized'].apply(
                    lambda x: [wnl.lemmatize(word[0], word[2]) for word in x])
                
                # BERT-Base
                df['_BERT-Tokenized-Base'] = df['BERT-Tokenized-Base'].apply(
                    lambda x: [w + tuple('a') for w in x if 'JJ' in w[1]])
                df['_BERT-Tokenized-Base'] += df['BERT-Tokenized-Base'].apply(
                    lambda x: [w + tuple('v') for w in x if 'VB' in w[1]])
                df['_BERT-Tokenized-Base'] += df['BERT-Tokenized-Base'].apply(
                    lambda x: [w + tuple('n') for w in x if 'NN' in w[1]])
                df['BERT-Tokenized-Base'] = df['_BERT-Tokenized-Base']
                df = df.drop(columns=['_BERT-Tokenized-Base'])
                df['BERT-Tokenized-Base'] = df['BERT-Tokenized-Base'].apply(
                    lambda x: [wnl.lemmatize(word[0], word[2]) for word in x])

                # BERT-Large
                df['_BERT-Tokenized-Large'] = df['BERT-Tokenized-Large'].apply(
                    lambda x: [w + tuple('a') for w in x if 'JJ' in w[1]])
                df['_BERT-Tokenized-Large'] += df['BERT-Tokenized-Large'].apply(
                    lambda x: [w + tuple('v') for w in x if 'VB' in w[1]])
                df['_BERT-Tokenized-Large'] += df['BERT-Tokenized-Large'].apply(
                    lambda x: [w + tuple('n') for w in x if 'NN' in w[1]])
                df['BERT-Tokenized-Large'] = df['_BERT-Tokenized-Large']
                df = df.drop(columns=['_BERT-Tokenized-Large'])
                df['BERT-Tokenized-Large'] = df['BERT-Tokenized-Large'].apply(
                    lambda x: [wnl.lemmatize(word[0], word[2]) for word in x])
                print(SIMP+fblue+bgray+'\n Lemmatization Done!'+End)
                
            
            # Formatting for BERT
            elif item[i] == 'BERT-Format':
                # Base
                df['BERT-Tokenized-Base'] = df['BERT-Tokenized-Base'].apply(lambda x: ['[CLS]'] + x + ['[SEP]'])
                df['TokenID-Base'] = df['BERT-Tokenized-Base'].apply(Btokenizer.convert_tokens_to_ids)
                df['SegmentID-Base'] = df['BERT-Tokenized-Base'].apply(lambda x: [1]*len(x))
                df['Token-Tensor-Base'] = df['TokenID-Base'].apply(lambda x: torch.tensor([x]))
                df['Segment-Tensor-Base'] = df['SegmentID-Base'].apply(lambda x: torch.tensor([x]))

                # Large
                df['BERT-Tokenized-Large'] = df['BERT-Tokenized-Large'].apply(lambda x: ['[CLS]'] + x + ['[SEP]'])
                df['TokenID-Large'] = df['BERT-Tokenized-Large'].apply(Ltokenizer.convert_tokens_to_ids)
                df['SegmentID-Large'] = df['BERT-Tokenized-Large'].apply(lambda x: [1]*len(x))
                df['Token-Tensor-Large'] = df['TokenID-Large'].apply(lambda x: torch.tensor([x]))
                df['Segment-Tensor-Large'] = df['SegmentID-Large'].apply(lambda x: torch.tensor([x]))
                print(SIMP+fblue+bgray+'\n Data Prepared For BERT!\n'+End)
            

    return df







def Concat(df: pd.DataFrame, col: str, i: int):
    if (len(df)-1) == i:
        return df[col].iloc[i]
    return np.concatenate((df[col].iloc[i], Concat(df, col, i+1)), axis=0)







def Split(arr: np.array, df: pd.DataFrame, col: str):

    l = []
    pos = 0
    for i in range(len(df)):
        l.append(list(arr[pos:len(df[col].iloc[i]) + pos]))
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
                
                if df[Tagged_Sentences].iloc[i][j] in fd[df[Vectors_Clusters].iloc[i][j]]:
                    fd[df[Vectors_Clusters].iloc[i][j]][df[Tagged_Sentences].iloc[i][j]] += 1
                else:
                    fd[df[Vectors_Clusters].iloc[i][j]][df[Tagged_Sentences].iloc[i][j]] = 1

            else:
                fd[df[Vectors_Clusters].iloc[i][j]] = {df[Tagged_Sentences].iloc[i][j] : 1}

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
                ds[df[Vectors_Clusters].iloc[i][j]] = [[df[IDs].iloc[i][j], df[Coordinates].iloc[i][j]]]

    cc = pd.read_csv('centers_clusters.csv')
    cc[Vectors_Clusters] = cc[Vectors_Clusters].apply(lambda x: eval(x))
    cc = cc[Vectors_Clusters].tolist()

    # CC[0]->Label 0
    cor = 1  # Coordinate index
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
            d[df[IDs].iloc[i][j]] = df[Tagged_Sentences].iloc[i][j]

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
        return 'neutral'
    elif pos > neg:
        return 'positive'
    else:
        return 'negative'