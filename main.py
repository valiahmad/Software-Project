from _.Color import *
print(ITALIC+fgray+borange+' Loading Libraries...'+End)
import pandas as pd
from preprocessing import Preprocess, Concat, Split, setID, Polarity, evalFeature
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
df = df[['SenID', 'Review', 'Feature', 'Polarity', 'Category', 'Tokenized',
       'BERT-Tokenized-Base', 'BERT-Tokenized-Large', 'Bb-S-K', 'Bb-t-K', 'Bl-S-K', 'Bl-t-K',
       'W-S-K', 'W-t-K', 'IDs-NBERT', 'IDs-BERT-B', 'IDs-BERT-L', 'Sentiment-Predicted'
       ]]
############################################################################

# TODO: Evaluation
from sklearn.metrics import classification_report
df['trueLabel'] = pd.Categorical(df['Polarity']).codes
df['predLabel'] = pd.Categorical(df['Sentiment-Predicted']).codes
# Laptops
print(SIMP+forange+bblue+'\nFor Laptops'+End)
print(classification_report(df.loc[df['Category'] == 'Laptops', 'trueLabel'], 
                            df.loc[df['Category'] == 'Laptops', 'predLabel'], 
                            zero_division=1))
dfevL = pd.DataFrame(classification_report(df.loc[df['Category'] == 'Laptops', 'trueLabel'], 
                                          df.loc[df['Category'] == 'Laptops', 'predLabel'], 
                                          output_dict=True, 
                                          zero_division=1))
dfevL.to_excel('./Report/evaluation-sentiment-analysis-Laptops.xlsx', index=False)
# Restaurant
print(SIMP+forange+bblue+'\nFor Restaurant'+End)
print(classification_report(df.loc[df['Category'] == 'Restaurant', 'trueLabel'], 
                            df.loc[df['Category'] == 'Restaurant', 'predLabel'], 
                            zero_division=1))
dfevR = pd.DataFrame(classification_report(df.loc[df['Category'] == 'Restaurant', 'trueLabel'], 
                                          df.loc[df['Category'] == 'Restaurant', 'predLabel'], 
                                          output_dict=True, 
                                          zero_division=1))
dfevR.to_excel('./Report/evaluation-sentiment-analysis-Restaurant.xlsx', index=False)

totalTrueAspects = len(df['Feature'])                          # denominator -> recall

correctExtractedAspects_BbSK = evalFeature(df, dfPA, 
                                           ['IDs-BERT-B', 'BERT-Tokenized-Base', 'Bb-S-K'])   # numerator
correctExtractedAspects_BbtK = evalFeature(df, dfPA, 
                                           ['IDs-BERT-B', 'BERT-Tokenized-Base', 'Bb-t-K'])
correctExtractedAspects_BlSK = evalFeature(df, dfPA, 
                                           ['IDs-BERT-L', 'BERT-Tokenized-Large', 'Bl-S-K'])
correctExtractedAspects_BltK = evalFeature(df, dfPA, 
                                           ['IDs-BERT-L', 'BERT-Tokenized-Large', 'Bl-t-K'])
correctExtractedAspects_WSK = evalFeature(df, dfPA, 
                                           ['IDs-NBERT', 'Tokenized', 'W-S-K'])
correctExtractedAspects_WtK = evalFeature(df, dfPA, 
                                           ['IDs-NBERT', 'Tokenized', 'W-t-K'])

totalCorrectExtractedAspects_BbSk = len(df['Bb-S-K'])          # denominator -> precision
totalCorrectExtractedAspects_Bbtk = len(df['Bb-t-K'])
totalCorrectExtractedAspects_BlSk = len(df['Bl-S-K'])
totalCorrectExtractedAspects_Bltk = len(df['Bl-t-K'])
totalCorrectExtractedAspects_WSk = len(df['W-S-K'])
totalCorrectExtractedAspects_Wtk = len(df['W-t-K'])

metricsFeature = {'Precision':{
    'Bb-S-K':correctExtractedAspects_BbSK / totalCorrectExtractedAspects_BbSk, 
    'Bb-t-K':correctExtractedAspects_BbtK / totalCorrectExtractedAspects_Bbtk,
    'Bl-S-K':correctExtractedAspects_BlSK / totalCorrectExtractedAspects_BlSk, 
    'Bl-t-K':correctExtractedAspects_BltK / totalCorrectExtractedAspects_Bltk, 
    'W-S-K':correctExtractedAspects_WSK / totalCorrectExtractedAspects_WSk, 
    'W-t-K':correctExtractedAspects_WtK / totalCorrectExtractedAspects_Wtk
 },
 'Recall':{
     'Bb-S-K':correctExtractedAspects_BbSK / totalTrueAspects, 
     'Bb-t-K':correctExtractedAspects_BbtK / totalTrueAspects,
     'Bl-S-K':correctExtractedAspects_BlSK / totalTrueAspects, 
     'Bl-t-K':correctExtractedAspects_BltK / totalTrueAspects, 
     'W-S-K':correctExtractedAspects_WSK / totalTrueAspects, 
     'W-t-K':correctExtractedAspects_WtK / totalTrueAspects
 }}
metricsFeature['F1-Score'] = {
    'Bb-S-K':2 * metricsFeature['Precision']['Bb-S-K'] * metricsFeature['Recall']['Bb-S-K'] / 
                (metricsFeature['Precision']['Bb-S-K'] + metricsFeature['Recall']['Bb-S-K']), 
    'Bb-t-K':2 * metricsFeature['Precision']['Bb-t-K'] * metricsFeature['Recall']['Bb-t-K'] / 
                (metricsFeature['Precision']['Bb-t-K'] + metricsFeature['Recall']['Bb-t-K']), 
    'Bl-S-K':2 * metricsFeature['Precision']['Bl-S-K'] * metricsFeature['Recall']['Bl-S-K'] / 
                (metricsFeature['Precision']['Bl-S-K'] + metricsFeature['Recall']['Bl-S-K']), 
    'Bl-t-K':2 * metricsFeature['Precision']['Bl-t-K'] * metricsFeature['Recall']['Bl-t-K'] / 
                (metricsFeature['Precision']['Bl-t-K'] + metricsFeature['Recall']['Bl-t-K']), 
    'W-S-K': 2 * metricsFeature['Precision']['W-S-K'] * metricsFeature['Recall']['W-S-K'] / 
                (metricsFeature['Precision']['W-S-K'] + metricsFeature['Recall']['W-S-K']), 
    'W-t-K': 2 * metricsFeature['Precision']['W-t-K'] * metricsFeature['Recall']['W-t-K'] / 
                (metricsFeature['Precision']['W-t-K'] + metricsFeature['Recall']['W-t-K'])
}

# TODO: GUI