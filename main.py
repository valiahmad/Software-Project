from settings import set


Settings = set()

if Settings['Punctuation']:
    print('Punctuation Done!')
else:
    print('Punctuation Failed!')

if Settings['Digit']:
    print('Digit Done!')
else:
    print('Digit Failed!')

if Settings['<10letters']:
    print('<10letters Done!')
else:
    print('<10letters Failed!')

if Settings['Tokenization']:
    print('Tokenization Done!')
else:
    print('Tokenization Failed!')

if Settings['StopWords']:
    print('StopWords Done!')
else:
    print('StopWords Failed!')

if Settings['POS Tagging']:
    print('POS Tagging Done!')
else:
    print('POS Tagging Failed!')

if Settings['Report']:
    print('Report Done!')
else:
    print('Report Failed!')

if Settings['Spell Checking']:
    print('Spell Checking Done!')
else:
    print('Spell Checking Failed!')

if Settings['Stemming']:
    print('Stemming Done!')
else:
    print('Stemming Failed!')

if Settings['Lemmatization']:
    print('Lemmatization Done!')
else:
    print('Lemmatization Failed!')
