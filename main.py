from _.Color import *
print(ITALIC+fgray+borange+' Loading Libraries...'+End)
import pandas as pd
from preprocessing import Preprocess, Concat, Split, setID, Polarity
from word_embedding import wordEmbed
from dimensionality_reduction import dimReduc
from clustering import Cluster
from feature_selection import FeatureSelector
print(ITALIC+fwhite+bgreen_yashmi+'\n Loading Done!'+End)
from _.settings import set
Settings = set()



df = Preprocess()

df = wordEmbed(df)

df = pd.concat([dimReduc(df['BERT'], 'BERT'), df], axis=1)

df = pd.concat([dimReduc(df['Word2Vec'], 'Word2Vec'), df], axis=1)

df['B-S-k'] = Split(Cluster(Concat(df['SOM_BERT'], 'SOM_BERT', 0), 'B-S-K'), df['SOM_BERT'], 'SOM_BERT')

df['B-t-K'] = Split(Cluster(Concat(df['tSNE_BERT'], 'tSNE_BERT', 0), 'B-t-K'), df['tSNE_BERT'], 'tSNE_BERT')

df['W-S-K'] = Split(Cluster(Concat(df['SOM_Word2Vec'], 'SOM_Word2Vec', 0), 'W-S-K'), df['SOM_Word2Vec'], 'SOM_Word2Vec')

df['W-t-K'] = Split(Cluster(Concat(df['tSNE_Word2Vec'], 'tSNE_Word2Vec', 0), 'W-t-K'), df['tSNE_Word2Vec'], 'tSNE_Word2Vec')

df['IDs-NBERT'] = setID(df['Tokenized'], 'Tokenized')

df['IDs-BERT'] = setID(df['BERT-Tokenized'], 'BERT-Tokenized')

df = df[['Review', 'Tokenized', 'Tokenized-POST', 'BERT-Tokenized-POST', 'IDs-NBERT', 'IDs-BERT', 'SOM_BERT',
         'tSNE_BERT', 'SOM_Word2Vec', 'tSNE_Word2Vec', 'B-S-k', 'B-t-K', 'W-S-K', 'W-t-K']]

BSK = ['BERT-Tokenized-POST', 'IDs-BERT', 'SOM_BERT', 'B-S-k']
Prominent_Aspects_BSK = FeatureSelector(df[BSK], BSK)

BtK = ['BERT-Tokenized-POST', 'IDs-BERT', 'tSNE_BERT', 'B-t-K']
Prominent_Aspects_BtK = FeatureSelector(df[BtK], BtK)

WSK = ['Tokenized-POST', 'IDs-NBERT', 'SOM_Word2Vec', 'W-S-K']
Prominent_Aspects_WSK = FeatureSelector(df[WSK], WSK)

WtK = ['Tokenized-POST', 'IDs-NBERT', 'tSNE_Word2Vec', 'W-t-K']
Prominent_Aspects_WtK = FeatureSelector(df[WtK], WtK)

df['Sentiment-Predicted'] = df['Tokenized'].apply(lambda x: Polarity(x))