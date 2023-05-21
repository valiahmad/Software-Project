from _.Color import *
def setInit():
    setList = {'Lowercase':0, 'Punctuation':0, 'Digit':0, '<10letters':0,
               'Tokenization':0, 'BERT-Tokenization':0, 'Set IDs':0,
               'Feature ID':0, 'Spell Checking':0, 'POS Tagging':0,
               'StopWords':0, 'Stemming':0, 'Lemmatization':0,
               'BERT-Format':0
               }
    option = ['Lowercase', 'Punctuation', 'Digit', '<10letters',
              'Tokenization', 'BERT-Tokenization', 'Set IDs',
              'Feature ID', 'Spell Checking', 'POS Tagging', 'StopWords',
              'Stemming', 'Lemmatization', 'BERT-Format'
              ]
    
    while True:

        print(SIMP+fgreen_yashmi+bgray+'\n*To set the procedures of preprocessing \nfor each item enter the numbers in order and press Enter. (Ex.: 2,4,1,3,...)'+End)
        for i in range(len(option)):
            print(i+1,'.',option[i])
        s = input()
        s = s.split(',')
        for i in range(len(s)):
            setList[option[int(s[i])-1]] = i+1
        
        setList = {val:key for key, val in setList.items()}

        print(SIMP+fgreen_yashmi+bgray+'The order is set: '+End)
        for i in range(1, len(setList)+1):
            if i in setList:
                print(ITALIC+forange+bgray+str(i)+'.'+setList[i]+End)
        
        e = input('To edit press 0\nTo continue press any key...')
        if e != '0':
            break


    return setList