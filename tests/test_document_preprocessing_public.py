import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from document_preprocessor import SplitTokenizer, RegexTokenizer, SpaCyTokenizer

mwe_filepath = '/Users/linfangqing/Desktop/SI650/HW/HW1/starter-code/tests/multi_word_expressions.txt'
mwe_list = []
with open(mwe_filepath, 'r') as f: 
    lines = f.readlines()
    for line in lines:
        mwe_list.append(line.strip())

############ =======Test SplitTokenizer=========== ############
class TestSplitTokenizer(unittest.TestCase):
    """Test SplitTokenizer."""
    def test_split_empty_doc(self):
        """Test tokenizing an empty document."""
        text = ""
        expected_tokens = []
        tokenizer = SplitTokenizer(multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_one_token_doc(self):
        """Test tokenizing an one-token document."""
        text = "Michigan"
        expected_tokens = ["michigan"]
        tokenizer = SplitTokenizer(multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_doc_with_punc(self):
        """Test tokenizing a document with punctuations."""
        text = "This is a test sentence."
        expected_tokens = ['This', 'is', 'a', 'test', 'sentence.']
        tokenizer = SplitTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)
        
    def test_unicode_text(self):
        """Test tokenizing a text with an emoji."""
        text = "Welcome to the United States 🫥"
        expected_tokens = ['Welcome', 'to', 'the', 'United States', '🫥']
        tokenizer = SplitTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_MWE_recognition(self):
        """Test tokenizing a document with multi-word expressions."""
        text = "The United Nations Development Programme is a United Nations agency."
        expected_tokens = ['The', 'United Nations Development Programme', 'is', 'a', 'United Nations', 'agency.']
        tokenizer = SplitTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_MWE_with_punc(self):
        """Test tokenizing a document with a multi-word expression
           containing a punctuation."""
        # TODO - Implement this test
        # You are only required to implement the test case, which we will use for grading. 
        # Nevertheless, we encourage you to implement additional test cases to ensure your code is correct. 
        text = "The United Nations Development Programme is, .. , a United Nations agency."
        expected_tokens = ['The', 'United Nations Development Programme', 'is,', '..', ',', 'a', 'United Nations', 'agency.']
        tokenizer = SplitTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)
        
    def test1(self):
        """Test tokenizing a document with a multi-word expression
           containing a punctuation."""
        # TODO - Implement this test
        # You are only required to implement the test case, which we will use for grading. 
        # Nevertheless, we encourage you to implement additional test cases to ensure your code is correct. 
        text = "The United Nations is  United Nations Development Programme."
        expected_tokens = ['The', 'United Nations', 'is', 'United Nations Development Programme', '.']
        tokenizer = SplitTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test2(self):
        """Test tokenizing a document with a multi-word expression
           containing a punctuation."""
        # TODO - Implement this test
        # You are only required to implement the test case, which we will use for grading. 
        # Nevertheless, we encourage you to implement additional test cases to ensure your code is correct. 
        text = "The United Nations is  Mexican-Kenyan actress Charity: Water."
        expected_tokens = ['The', 'United Nations', 'is', "Mexican-Kenyan", 'actress', 'Charity: Water', '.']
        tokenizer = SplitTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)
        
############ =======Test RegexTokenizer=========== ############
class TestRegexTokenizer(unittest.TestCase):
    """Test RegexTokenizer."""
    def test_split_empty_doc(self):
        """Test tokenizing an empty document."""
        text = ""
        expected_tokens = []
        tokenizer = RegexTokenizer(multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_one_token_doc(self):
        """Test tokenizing an one-token document."""
        text = "Michigan"
        expected_tokens = ["michigan"]
        tokenizer = RegexTokenizer(multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_doc_with_punc(self):
        """Test tokenizing a document with punctuations."""
        text = "Hello, my name is Tomato Lin, and I keep a cute cat: 'little ghost'!"
        expected_tokens = ['Hello', 'my', 'name', 'is', 'Tomato', 'Lin', 'and', 'I', 'keep', 'a', 'cute', 'cat', 'little', 'ghost']
        tokenizer = RegexTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_MWE_recognition(self):
        """Test tokenizing a text with an emoji."""
        text = "Welcome to the United States 🫥"
        expected_tokens = ['Welcome', 'to', 'the', 'United States']
        tokenizer = RegexTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)


############ =======Test SpaCyTokenizer=========== ############
class TestSpaCyTokenizer(unittest.TestCase):
    """Test SpaCyTokenizer."""
    def test_split_empty_doc(self):
        """Test tokenizing an empty document."""
        text = ""
        expected_tokens = []
        tokenizer = SpaCyTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_one_token_doc(self):
        """Test tokenizing an one-token document."""
        text = "Michigan"
        expected_tokens = ["Michigan"]
        tokenizer = SpaCyTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_MWE_recognition(self):
        """Test tokenizing a document with multi-word expressions."""
        text = "UNICEF, now officially United Nations Children's Fund, is a United Nations agency."
        expected_tokens = ['UNICEF', ',', 'now', 'officially', "United Nations Children's Fund", ',', 'is', 'a', 'United Nations', 'agency', '.']
        tokenizer = SpaCyTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_unicode_text(self):
        """Test tokenizing a text with an emoji."""
        text = "Welcome to America 🫥"
        expected_tokens = ['Welcome', 'to', 'America', '🫥']
        tokenizer = SpaCyTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_MWE_not_recognized_by_spacy(self):
        """Test tokenizing a document with multi-word expressions
           not recognized by SpaCy."""
        text = "Lupita Nyong'o is a Mexican-Kenyan actress."
        expected_tokens = ["Lupita Nyong'o", 'is', 'a', 'Mexican', '-', 'Kenyan', 'actress', '.']
        tokenizer = SpaCyTokenizer(lowercase=False, multiword_expressions=mwe_list)
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)


if __name__ == '__main__':
    unittest.main()
