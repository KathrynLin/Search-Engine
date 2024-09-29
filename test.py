mwe_filepath = '/Users/linfangqing/Desktop/SI650/HW/HW1/starter-code/tests/multi_word_expressions.txt'
mwe_list = []
with open(mwe_filepath, 'r') as f: 
    lines = f.readlines()
    for line in lines:
        mwe_list.append(line.strip())
        
mwe_list.sort(key=len, reverse=True)
multiword_expressions_tokens = [token.split() for token in mwe_list]
#print(multiword_expressions_tokens)
for phrase in multiword_expressions_tokens:
    #print(phrase)
    pass

import string
if "'" in string.punctuation:
    print("yes")
else:
    print("no")