from Color import *
print(ITALIC+fgray+borange+'\n Loading Libraries...'+End)
################################ LIBRARIES
from settings import setting
import string
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


# Loading Data
df = Parse()
df = df.returnMode(dataFrameMode=True)
print(df.columns)

# Punctuations Removal !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
if Settings['Punctuation']:
    df['Review'] = df['Review'].str.translate(str.maketrans('', '', string.punctuation))

# Digit Removal
if Settings['Digit']:
    df['Review'] = df['Review'].str.translate(str.maketrans('', '', string.digits))

# Reviews Removal of less than 10 letters
if Settings['<10letters']:
    df = df.drop(df[df['Review'].str.len() < 10].index)

# Tokenized reviews text
if Settings['Tokenization']:
    df['Tokenized'] = df['Review'].apply(word_tokenize)
else:
    print(BOLD+fred+bgray+'\nData is NOT Tokenized!!!'+End)
    input('\nTo Continue Press Enter...')

# Correcting word spells
if Settings['Spell Checking']:
    spell = SpellChecker()
    df['Tokenized'] = df['Tokenized'].apply(lambda x: [spell.correction(word) for word in x])
else:
    print(BOLD+fred+bgray+'\nData is NOT Spell-Checked!!!'+End)
    input('\nTo Continue Press Enter...')

# Tagging Part-of-Speech
if Settings['POS Tagging']:
    df['Tagged'] = df['Tokenized'].apply(pos_tag)

# StopWords Removing
if Settings['StopWords']:
    stop_words = set(stopwords.words('english'))
    df['StopWordsRemoved'] = df['Tokenized'].apply(
        lambda x: [word for word in x if word.lower() not in stop_words])

# Stemming
if Settings['Stemming']:
    stemmer = SnowballStemmer("english")
    df['Stemmed'] = df['Tokenized'].apply(lambda x: [stemmer.stem(word) for word in x])

#Lemmatization
if Settings['Lemmatization']:
    wnl = WordNetLemmatizer()
    df['Lemmatized'] = df['Tokenized'].apply(lambda x: [wnl.lemmatize(word, 'v') for word in x])
