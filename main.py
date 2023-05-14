from _.Color import *
print(ITALIC+fgray+borange+' Loading Libraries...'+End)
import pandas as pd
from preprocessing import Preprocess, Concat, Split
from word_embedding import wordEmbed
from dimensionality_reduction import dimReduc
from clustering import Cluster
print(ITALIC+fwhite+bgreen_yashmi+'\n Loading Done!'+End)
from _.settings import set
Settings = set()



df = Preprocess()

df = wordEmbed(df)

df = pd.concat([dimReduc(df['BERT'], 'BERT'), df], axis=1)

df = pd.concat([dimReduc(df['Word2Vec'], 'Word2Vec'), df], axis=1)

df['B-S-k'] = Split(Cluster(Concat(df['SOM_BERT'], 'SOM_BERT', 0)), df['SOM_BERT'], 'SOM_BERT')

df['B-t-K'] = Split(Cluster(Concat(df['tSNE_BERT'], 'tSNE_BERT', 0)), df['tSNE_BERT'], 'tSNE_BERT')

df['W-S-K'] = Split(Cluster(Concat(df['SOM_Word2Vec'], 'SOM_Word2Vec', 0)), df['SOM_Word2Vec'], 'SOM_Word2Vec')

df['W-t-K'] = Split(Cluster(Concat(df['tSNE_Word2Vec'], 'tSNE_Word2Vec', 0)), df['tSNE_Word2Vec'], 'tSNE_Word2Vec')
