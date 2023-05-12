from _.Color import *
print(BOLD+fgray+bwhite+' Preprocessing'+End)

import string
import torch
import pandas as pd
from _.settings import setting
from _.datasetToDict import Parse
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from spellchecker import SpellChecker
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Reading the settings
Settings = setting()



def Preprocess():

    # Loading Data
    if Settings['']:
        df = Parse()
        df = df.returnMode(dataFrameMode=True)
        print(df.head())
    else:
        path = './Datasets\\Laptops\\train.json'
        df = pd.read_json(path)
        df.head()

    

    # To Lowercase
    if Settings['Lowercase']:
        df['Review'] = df['Review'].str.lower()
        
    # Punctuations Removal !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    if Settings['Punctuation']:
        df['Review'] = df['Review'].str.translate(str.maketrans('', '', string.punctuation))
        print(SIMP+fblue+bgray+'\n Punctuation Removal Done!'+End)

    # Digit Removal
    if Settings['Digit']:
        df['Review'] = df['Review'].str.translate(str.maketrans('', '', string.digits))
        print(SIMP+fblue+bgray+'\n Digit Removal Done!'+End)

    # Reviews Removal of less than 10 letters
    if Settings['<10letters']:
        df = df.drop(df[df['Review'].str.len() < 10].index)
        print(SIMP+fblue+bgray+'\n Less Than 10 Letters Reviews Droped!'+End)

    # Tokenized reviews text
    if Settings['Tokenization']:
        df['Tokenized'] = df['Review'].apply(word_tokenize)
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

    # Correcting word spells
    if Settings['Spell Checking']:
        spell = SpellChecker()
        df['Tokenized-Original'] = df['Tokenized']
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
        df['StopWordsRemoved'] = df['Tokenized'].apply(
            lambda x: [word for word in x if word.lower() not in stop_words])
        print(SIMP+fblue+bgray+'\n StopWords Removal Done!'+End)

    # Stemming
    if Settings['Stemming']:
        stemmer = SnowballStemmer("english")
        df['Stemmed'] = df['Tokenized'].apply(lambda x: [stemmer.stem(word) for word in x])
        print(SIMP+fblue+bgray+'\n Words Stemmed!'+End)

    # Lemmatization
    if Settings['Lemmatization']:
        wnl = WordNetLemmatizer()
        df['Lemmatized'] = df['Tokenized'].apply(lambda x: [wnl.lemmatize(word, 'v') for word in x])
        print(SIMP+fblue+bgray+'\n Lemmatization Done!'+End)

    # Fixing the problems with dataset
    if Settings['Fix Problem']:

        print(SIMP+fblue+bgray+'\n Problems With The Dataset Fixed!'+End)
    
    # Formatting for BERT
    if Settings['BERT Format']:
        if Settings['Tokenization']:
            df['Tokenized'] = df['Tokenized'].apply(lambda x: ['[CLS]'] + x + ['[SEP]'])
            df['TokenID'] = df['Tokenized'].apply(tokenizer.convert_tokens_to_ids)
            df['SegmentID'] = df['Tokenized'].apply(lambda x: [1]*len(x))
            
        elif Settings['BERT-Tokenization']:
            df['BERT-Tokenized'] = df['BERT-Tokenized'].apply(lambda x: ['[CLS]'] + x + ['[SEP]'])
            df['TokenID'] = df['BERT-Tokenized'].apply(tokenizer.convert_tokens_to_ids)
            df['SegmentID'] = df['BERT-Tokenized'].apply(lambda x: [1]*len(x))
            
        df['Token-Tensor'] = df['TokenID'].apply(lambda x: torch.tensor([x]))
        df['Segment-Tensor'] = df['SegmentID'].apply(lambda x: torch.tensor([x]))
        print(SIMP+fblue+bgray+'\n Data Prepared For BERT!'+End)
    
    

    return df
