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



df = Preprocess(procedures)
print(ITALIC+fwhite+bgreen_yashmi+'Preprocess Done!'+End)

df = wordEmbed(df)
print(ITALIC+fwhite+bgreen_yashmi+'\n wordEmbed Done!'+End)

##################################################################### Dimensionality Reduction
df = pd.concat([dimReduc(df, 'BERT-Base'), df], axis=1)
print(ITALIC+fwhite+bgreen_yashmi+'\n dimReduc BERT-Base Done!'+End)

df = pd.concat([dimReduc(df, 'BERT-Large'), df], axis=1)
print(ITALIC+fwhite+bgreen_yashmi+'\n dimReduc BERT-Large Done!'+End)

df = pd.concat([dimReduc(df, 'Word2Vec'), df], axis=1)
print(ITALIC+fwhite+bgreen_yashmi+'\n dimReduc Word2Vec Done!'+End)
############################################################################ Clustring
df['Bb-S-K'] = Split(Cluster(Concat(df, 'SOM_BERT-Base', 0), 'Bb-S-K'), df, 'SOM_BERT-Base')
print(ITALIC+fwhite+bgreen_yashmi+'\n Clustring Bb-S-K Done!'+End)

df['Bb-t-K'] = Split(Cluster(Concat(df, 'tSNE_BERT-Base', 0), 'Bb-t-K'), df, 'tSNE_BERT-Base')
print(ITALIC+fwhite+bgreen_yashmi+'\n Clustring Bb-t-K Done!'+End)

df['Bl-S-K'] = Split(Cluster(Concat(df, 'SOM_BERT-Large', 0), 'Bl-S-K'), df, 'SOM_BERT-Large')
print(ITALIC+fwhite+bgreen_yashmi+'\n Clustring Bl-S-K Done!'+End)

df['Bl-t-K'] = Split(Cluster(Concat(df, 'tSNE_BERT-Large', 0), 'Bl-t-K'), df, 'tSNE_BERT-Large')
print(ITALIC+fwhite+bgreen_yashmi+'\n Clustring Bl-t-K Done!'+End)

df['W-S-K'] = Split(Cluster(Concat(df, 'SOM_Word2Vec', 0), 'W-S-K'), df, 'SOM_Word2Vec')
print(ITALIC+fwhite+bgreen_yashmi+'\n Clustring W-S-K Done!'+End)

df['W-t-K'] = Split(Cluster(Concat(df, 'tSNE_Word2Vec', 0), 'W-t-K'), df, 'tSNE_Word2Vec')
print(ITALIC+fwhite+bgreen_yashmi+'\n Clustring W-t-K Done!'+End)
############################################################ Set ID for each token
df['IDs-NBERT'] = setID(df, 'Tokenized')
print(ITALIC+fwhite+bgreen_yashmi+'\n IDs-NBERT Done!'+End)

df['IDs-BERT-B'] = setID(df, 'BERT-Tokenized-Base')
print(ITALIC+fwhite+bgreen_yashmi+'\n IDs-BERT-B Done!'+End)

df['IDs-BERT-L'] = setID(df, 'BERT-Tokenized-Large')
print(ITALIC+fwhite+bgreen_yashmi+'\n IDs-BERT-L Done!'+End)

#############################################################
df.to_excel('./Report/report-1.xlsx', index=False)
print(ITALIC+fwhite+bgreen_yashmi+'\n report-1 Done!'+End)
df = df[['SenID', 'Review', 'Feature', 'Polarity', 'Category', 'Tokenized',
       'BERT-Tokenized-Base', 'BERT-Tokenized-Large',
       'SOM_BERT-Base', 'tSNE_BERT-Base', 'SOM_BERT-Large', 'tSNE_BERT-Large',
       'SOM_Word2Vec', 'tSNE_Word2Vec', 'Bb-S-K', 'Bb-t-K', 'Bl-S-K', 'Bl-t-K',
       'W-S-K', 'W-t-K', 'IDs-NBERT', 'IDs-BERT-B', 'IDs-BERT-L'
       ]]
#############################################################

####################################################################### Feature Extraction
BbSK = ['BERT-Tokenized-Base', 'IDs-BERT-B', 'SOM_BERT-Base', 'Bb-S-K']
Prominent_Aspects_BbSK = FeatureSelector(df, BbSK)
print(ITALIC+fwhite+bgreen_yashmi+'\n Prominent_Aspects_BbSK Done!'+End)

BbtK = ['BERT-Tokenized-Base', 'IDs-BERT-B', 'tSNE_BERT-Base', 'Bb-t-K']
Prominent_Aspects_BbtK = FeatureSelector(df, BbtK)
print(ITALIC+fwhite+bgreen_yashmi+'\n Prominent_Aspects_BbtK Done!'+End)

BlSK = ['BERT-Tokenized-Large', 'IDs-BERT-L', 'SOM_BERT-Large', 'Bl-S-K']
Prominent_Aspects_BlSK = FeatureSelector(df, BlSK)
print(ITALIC+fwhite+bgreen_yashmi+'\n Prominent_Aspects_BlSK Done!'+End)

BltK = ['BERT-Tokenized-Large', 'IDs-BERT-L', 'tSNE_BERT-Large', 'Bl-t-K']
Prominent_Aspects_BltK = FeatureSelector(df, BltK)
print(ITALIC+fwhite+bgreen_yashmi+'\n Prominent_Aspects_BltK Done!'+End)

WSK = ['Tokenized', 'IDs-NBERT', 'SOM_Word2Vec', 'W-S-K']
Prominent_Aspects_WSK = FeatureSelector(df, WSK)
print(ITALIC+fwhite+bgreen_yashmi+'\n Prominent_Aspects_WSK Done!'+End)

WtK = ['Tokenized', 'IDs-NBERT', 'tSNE_Word2Vec', 'W-t-K']
Prominent_Aspects_WtK = FeatureSelector(df, WtK)
print(ITALIC+fwhite+bgreen_yashmi+'\n Prominent_Aspects_WtK Done!'+End)

dfPA = pd.DataFrame(columns=['Bb-S-K', 'Bb-t-K', 'Bl-S-K', 'Bl-t-K', 'W-S-K', 'W-t-K'],
                    data=[[Prominent_Aspects_BbSK, Prominent_Aspects_BbtK, Prominent_Aspects_BlSK, 
                           Prominent_Aspects_BltK, Prominent_Aspects_WSK, Prominent_Aspects_WtK]]
                    )
########################################################################## Sentiment Analysis
df['Sentiment-Predicted'] = df['Tokenized'].apply(lambda x: Polarity(x))
print(ITALIC+fwhite+bgreen_yashmi+'\n Sentiment-Predicted Done!'+End)
############################################################################
dfPA.to_excel('./Report/report-PA.xlsx', index=False)
df.to_excel('./Report/report-2-predicted-sentiment.xlsx', index=False)
print(ITALIC+fwhite+bgreen_yashmi+'\n report-PA&report-2 Done!'+End)
############################################################################

# TODO: Evaluation
correctExtractedAspects = 0
totalCorrectExtractedAspects = 0
totalTrueAspects = 0

# TODO: GUI