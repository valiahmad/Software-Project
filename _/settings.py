import json

def setting(setting = None):
    
    if setting:
        f = open('setting.ssm', 'w')
        json.dump(setting,f)
        f.close()
        
        
    elif not setting:
        f = open('setting.ssm')
        setting = json.load(f)
        f.close()
        

    return setting


listofSet = [\
    ['\nSelect Dataset : #1 Hu and Liu Datasets   #2 Stanford Datasets : ',\
    'Hu and Liu Datasets', True, False],\
    ['\nLowercase : #1 YES   #2 NO : ',\
    'Lowercase', True, False],\
    ['\nPunctuation Removal : #1 YES   #2 NO : ',\
    'Punctuation', True, False],\
    ['\nDigit Removal : #1 YES   #2 NO : ',\
    'Digit', True, False],\
    ['\nDo you want to remove reviews less than 10 letters : #1 YES   #2 NO : ',\
    '<10letters', True, False],\
    ['\nDo you want to tokenize(NLTK) : #1 YES   #2 NO : ',\
    'Tokenization', True, False],\
    ['\nDo you want to tokenize(BERT) : #1 YES   #2 NO : ',\
    'BERT-Tokenization', True, False],\
    ['\nStopWords : #1 YES   #2 NO : ',\
    'StopWords', True, False],\
    ['\nPOS Tagging : #1 YES   #2 NO : ',\
    'POS Tagging', True, False],\
    ['\nSpell Checking : #1 YES   #2 NO : ',\
     'Spell Checking', True, False],\
    ['\nStemming : #1 YES   #2 NO : ',\
     'Stemming', True, False],\
    ['\nLemmatization : #1 YES   #2 NO : ',\
     'Lemmatization', True, False],\
    ['\Formatting for BERT : #1 Prepare   #2 Don\'t : ',\
     'BERT Format', True, False],\
    ['\nFix Problem : #1 Active   #2 Deactive : ',\
     'Fix Problem', True, False],\
    ['\nDo You Want To Show A Report : #1 YES   #2 NO : ',\
    'Report', True, False]
    ]

def set():
    i = 0
    S = dict()
    while True:
        option = input(listofSet[i][0])
        if option == '1':
            S[listofSet[i][1]] = listofSet[i][2]
            i += 1
        elif option == '2':
            S[listofSet[i][1]] = listofSet[i][3]
            i += 1
        elif option == '3':
            S[listofSet[i][1]] = listofSet[i][4]
            i += 1
        elif option == '4':
            S[listofSet[i][1]] = listofSet[i][5]
            i += 1
        else:
            print('\nPlease Try Againg!!! ')
        if i == len(listofSet):
            break
    return setting(S)