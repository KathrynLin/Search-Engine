import tqdm
import json
import unittest
import sys
import os
import bisect
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# modules being tested
from document_preprocessor import RegexTokenizer
from indexing import IndexType, InvertedIndex, BasicInvertedIndex, Indexer, PositionalInvertedIndex
from collections import Counter, defaultdict

dataset_path = 'tests/dataset_2.jsonl'
max_docs = 4
document_preprocessor = RegexTokenizer('\w+')
text_key = 'text'
global_term_count = Counter()
doc_count = 0
minimum_word_frequency = 3
stopwords = ['this']
index = None
index_ls= {}
def filters(tokens, global_term_count, minimum_word_frequency, stopwords):
    filtered_tokens = [
        term if (term not in stopwords and global_term_count[term] >= minimum_word_frequency)
        else None
        for term in tokens
    ]
    return filtered_tokens
with open(dataset_path, 'r') as f:
    for line in f:
        if max_docs > 0 and doc_count >= max_docs:
            break 
        doc = json.loads(line)
        docid = int(doc['docid'])
        text = doc.get(text_key, '')
        tokens = document_preprocessor.tokenize(text)
        global_term_count.update(tokens)
        doc_count += 1
        index_ls[doc["docid"]] = tokens
        #print(tokens)
    for docid, tokens in index_ls.items():
        filtered_tokens = filters(tokens, global_term_count, minimum_word_frequency, stopwords)
        print(f"Original tokens for doc {docid}: {tokens}")
        print(f"Filtered tokens for doc {docid}: {filtered_tokens}")