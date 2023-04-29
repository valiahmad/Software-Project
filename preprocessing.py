from Color import *
print(BOLD+fgray+bwhite+' Preprocessing'+End)
print(ITALIC+fgray+borange+' Loading Libraries...'+End)
################################ LIBRARIES
import string
import pandas as pd
from settings import setting
from datasetToDict import Parse
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
################################ LIBRARIES
print(ITALIC+fwhite+bgreen_yashmi+'\n Loading Done!'+End)
Settings = setting()

def Preprocess(address):
    # Loading Data
    if Settings['Excel File']:
        df = pd.read_excel(address)
        print(SIMP+fblue+bgray+'\n Excel File Has Been Loaded!'+End)
    else:
        df = Parse()
        df = df.returnMode(dataFrameMode=True)
    print(df.columns)

    # Amount of Data to Use
    # df = df.sample(n=100)
    # if Settings['Percetage']:
    #     df = df.sample(frac=Settings['percetage'])
    #     print(SIMP+fblue+bgray+'\n %d%% Data Randomly Selected!'+End % Settings['percetage'])
    # elif Settings['Number']:
    #     df = df.sample(n=Settings['number'])
    #     print(SIMP+fblue+bgray+'\n %d Numbers Samples Randomly Selected!'+End % Settings['number'])
    # else:
    #     print(SIMP+fblue+bgray+'\n All Data Is Been Processed!'+End)


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

    # Correcting word spells
    if Settings['Spell Checking']:
        spell = SpellChecker()
        df['Tokenized-Original'] = df['Tokenized']
        df['Tokenized'] = df['Tokenized'].apply(lambda x: [spell.correction(word) for word in x])
        df['Token-Check'] = df['Tokenized'].apply(lambda x: [x for word in x if word is None])
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

    # Word Embedding
    # if Settings['']:

    #     print(SIMP+fblue+bgray+'\n Word Converted!'+End)


    return df
