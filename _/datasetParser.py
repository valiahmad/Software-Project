'''
***********************************************************************************************************
*    github.com/valiahmad (c)                                                                             *
*    April.17.2023                                                                                        *
*    Description:                                                                                         *
*        This code is written to parse the datasets of Hu and Liu* in a dictionary format, and return     *
*        as Dict or DataFrame or write in Excel.                                                          *
*        *Minqing Hu and Bing Liu, 2004. Department of Computer Sicence University of Illinios at Chicago *
***********************************************************************************************************
'''

import pandas as pd 
import _.DatasetsAddresses as DatasetsAddresses
dataTextAd = DatasetsAddresses.LName

data = {}       # containing the whole data in format : 
                # {index:{'Review':review context, 
                # 'Features':{'Name:feature name, 'Polarity':feature poloarity}, 
                # 'Feature':has or not}}


class Parse:
    
    def __init__(self):

        f = {}          # containing each review features
        w = ''          # containing review/feature name
        ft = ''         # containing feature type
        P = 0           # containing polarity of each feature
        F = 0           # flag to show feature name is loading
        FT = 0          # flag to show feature type is available
        CF = 0          # containing number of each review feature
        CR = 0          # containing number of reviews
        R = 0           # flag to show review context is loading
        G = 1           # flag to show nothing is loading
        Re = 0          # flag to show review context is loading(without feature)
        fl = []         # List of features of each review
        pl = []         # List of polarity of each review

        # parsing the whole 14 datasets at once and merging in one
        for ad in dataTextAd:

            print('\nLoading : {}...'.format(ad))

            dataString = open(ad, 'r',encoding='cp1252')
            S = dataString.read()

            for i in range(1, len(S)):
                if (S[i] >= 'A' and S[i] <= 'Z' or S[i] >= 'a' and S[i] <= 'z') and S[i-1] == '\n':
                    F = 1
                    G = 0
                    CF = 0

                if S[i] == '[' and F:
                    if S[i+1] == '+' or S[i+1] == '-':
                        P = 1
                        if S[i+1] == '-':
                            P = -P
                        if S[i+2] != ']':
                            P = int(S[i+2]) * P
                    else:
                        FT = 1
                        ft = S[i+1]

                if S[i] == ']' and F:
                    if S[i-1] >= '0' and S[i-1] <= '9':
                        CF += 1
                        f['f'+str(CF)] = {'Name':w.strip(), 'Polarity':P}
                        fl.append(w.strip())
                        pl.append(P)
                    if FT:
                        f['f'+str(CF)]['ft'+str(CF)] = ft
                        FT = 0
                        ft = ''

                    w = ''

                if S[i] == ',' and F:
                    P = 0
                    continue

                if S[i] == '#' and F:
                    F = 0
                    CF = 0
                    R = 1
                    P = 0
                    continue

                if S[i] == '\n' and R:
                    R = 0
                    G = 1
                    if 'ft1' in f:
                        data[CR] = {'Review':w[1:].strip(), 
                                    'Features':f, 'Feature':'Yes', 
                                    'Feature List':fl, 'Polarity List':pl, 
                                    'Feature Type':'Yes'}
                    else:
                        data[CR] = {'Review':w[1:].strip(), 
                                    'Features':f, 'Feature':'Yes', 
                                    'Feature List':fl, 'Polarity List':pl, 
                                    'Feature Type':'No'}
                    CR += 1
                    f = {}
                    fl = []
                    pl = []
                    w = ''

                if S[i] == '#' and S[i-1] == '\n':
                    G = 0
                    Re = 1
                    continue

                if S[i] == '\n' and Re:
                    G = 1
                    Re = 0
                    data[CR] = {'Review':w[1:].strip(), 'Feature':'No'}
                    CR += 1
                    w = ''

                if not P and not G:
                    w = w + S[i]


                print('\r[%-20s] %d%% ' % ('#' * (i//(len(S)//20)), i//(len(S)//20)*5), end = '')
            
        

    def writeToExcel(self, add, index=True, transpose=True):
        # writing to EXCEL
        df = pd.DataFrame(data)
        if transpose:
            df = df.T
        df = df.sort_values('Feature')
        df = df.applymap(lambda x: x.encode('unicode_escape').
                        decode('utf-8') if isinstance(x, str) else x)
        print(df.head)
        df.to_excel(add,index=index)

    def returnType(self, dictMode=False, dataFrameMode=True):
        if dataFrameMode:
            df = pd.DataFrame(data).T
            df = df[df['Feature List'].str.len() == 1]
            df = df[df['Feature Type'] == 'No']
            df['Polarity List'] = df['Polarity List'].apply(lambda x: x[0])
            df.loc[df['Polarity List'] > 0, 'Polarity'] = 'positive'
            df.loc[df['Polarity List'] < 0, 'Polarity'] = 'negative'
            df['Category'] = 'Hu&Liu'
            
            ########################################################
            path = [
            './Datasets\\Laptops\\train.json',
            './Datasets\\Laptops\\test.json',
            './Datasets\\Restaurants\\train.json',
            './Datasets\\Restaurants\\test.json'
            ]
            dfLtr = pd.read_json(path[0])
            dfLte = pd.read_json(path[1])
            dfRtr = pd.read_json(path[2])
            dfRte = pd.read_json(path[3])
            dfL = pd.concat([dfLtr, dfLte], ignore_index=True)
            dfR = pd.concat([dfRtr, dfRte], ignore_index=True)
            del dfLtr, dfLte, dfRtr, dfRte
            dfR['Category'] = 'Restaurant'
            dfL['Category'] = 'Laptops'
            
            dfLR = pd.concat([dfR, dfL], ignore_index=True)
            dfLR = dfLR[dfLR['aspects'].str.len() == 1]
            dfLR['Feature List'] = dfLR['aspects'].apply(lambda x: x[0]['term'])
            dfLR['Polarity'] = dfLR['aspects'].apply(lambda x: x[0]['polarity'])
            dfLR = dfLR[dfLR['Feature List'].str.len()==1]
            
            ########################################################
            df = pd.concat([dfLR, df], ignore_index=True)

            return df
        
        elif dictMode:
            return data