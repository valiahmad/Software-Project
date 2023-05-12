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
    ['\nRead Excel File : #1 YES   #2 NO : ',\
    'Excel File', True, False],\
    ['\nPunctuation Removal : #1 YES   #2 NO : ',\
    'Punctuation', True, False],\
    ['\nDigit Removal : #1 YES   #2 NO : ',\
    'Digit', True, False],\
    ['\nDo you want to remove reviews less than 10 letters : #1 YES   #2 NO : ',\
    '<10letters', True, False],\
    ['\nDo you want to tokenize : #1 YES   #2 NO : ',\
    'Tokenization', True, False],\
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