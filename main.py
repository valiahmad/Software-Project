from _.Color import *
print(ITALIC+fgray+borange+' Loading Libraries...'+End)
import pandas as pd
from preprocessing import Preprocess, Concat, Split, setID, Polarity
from word_embedding import wordEmbed
from dimensionality_reduction import dimReduc
from clustering import Cluster
from feature_selection import FeatureSelector
from _.settings import setInit
print(ITALIC+fwhite+bgreen_yashmi+'\n Loading Done!'+End)
procedures = setInit()

Labels =250
Threshold = 0
Top_Items = 0


df = Preprocess(procedures)

df = wordEmbed(df)

df = pd.concat([dimReduc(df['BERT-Base'], 'BERT-Base'), df], axis=1)

df = pd.concat([dimReduc(df['BERT-Large'], 'BERT-Large'), df], axis=1)

df = pd.concat([dimReduc(df['Word2Vec'], 'Word2Vec'), df], axis=1)

df['Bb-S-k'] = Split(Cluster(Concat(df['SOM_BERT-Base'], 'SOM_BERT-Base', 0), 'Bb-S-K'), df['SOM_BERT-Base'], 'SOM_BERT-Base')

df['Bb-t-K'] = Split(Cluster(Concat(df['tSNE_BERT-Base'], 'tSNE_BERT-Base', 0), 'Bb-t-K'), df['tSNE_BERT-Base'], 'tSNE_BERT-Base')

df['Bl-S-k'] = Split(Cluster(Concat(df['SOM_BERT-Large'], 'SOM_BERT-Large', 0), 'Bl-S-K'), df['SOM_BERT-Large'], 'SOM_BERT-Large')

df['Bl-t-K'] = Split(Cluster(Concat(df['tSNE_BERT-Large'], 'tSNE_BERT-Large', 0), 'Bl-t-K'), df['tSNE_BERT-Large'], 'tSNE_BERT-Large')

df['W-S-K'] = Split(Cluster(Concat(df['SOM_Word2Vec'], 'SOM_Word2Vec', 0), 'W-S-K'), df['SOM_Word2Vec'], 'SOM_Word2Vec')

df['W-t-K'] = Split(Cluster(Concat(df['tSNE_Word2Vec'], 'tSNE_Word2Vec', 0), 'W-t-K'), df['tSNE_Word2Vec'], 'tSNE_Word2Vec')

df['IDs-NBERT'] = setID(df['Tokenized'], 'Tokenized')

df['IDs-BERT-B'] = setID(df['BERT-Tokenized-Base'], 'BERT-Tokenized-Base')

df['IDs-BERT-L'] = setID(df['BERT-Tokenized-Large'], 'BERT-Tokenized-Large')

df = df[['Review', 'Tokenized', 'IDs-NBERT', 'IDs-BERT-B', 'IDs-BERT-L', 'SOM_BERT-Base',
         'tSNE_BERT-Base', 'SOM_BERT-Large', 'tSNE_BERT-Large', 'SOM_Word2Vec', 'tSNE_Word2Vec', 
         'Bb-S-k', 'Bb-t-K', 'Bl-S-k', 'Bl-t-K', 'W-S-K', 'W-t-K']]

BbSK = ['Tokenized', 'IDs-BERT-B', 'SOM_BERT-Base', 'Bb-S-k']
Prominent_Aspects_BSK = FeatureSelector(df[BbSK], BbSK)

BbtK = ['Tokenized', 'IDs-BERT-B', 'tSNE_BERT-Base', 'Bb-t-K']
Prominent_Aspects_BtK = FeatureSelector(df[BbtK], BbtK)

BlSK = ['Tokenized', 'IDs-BERT-L', 'SOM_BERT-Large', 'Bl-S-k']
Prominent_Aspects_BSK = FeatureSelector(df[BlSK], BlSK)

BltK = ['Tokenized', 'IDs-BERT-L', 'tSNE_BERT-Large', 'Bl-t-K']
Prominent_Aspects_BtK = FeatureSelector(df[BltK], BltK)

WSK = ['Tokenized', 'IDs-NBERT', 'SOM_Word2Vec', 'W-S-K']
Prominent_Aspects_WSK = FeatureSelector(df[WSK], WSK)

WtK = ['Tokenized', 'IDs-NBERT', 'tSNE_Word2Vec', 'W-t-K']
Prominent_Aspects_WtK = FeatureSelector(df[WtK], WtK)

df['Sentiment-Predicted'] = df['Tokenized'].apply(lambda x: Polarity(x))

# TODO: Evaluation

# TODO: GUI

# TODO: TEST...