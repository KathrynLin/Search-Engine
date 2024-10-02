"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques and build your own tokenizer.
"""

from nltk.tokenize import RegexpTokenizer
import spacy
import string
import re

class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # TODO: Save arguments that are needed as fields of this class
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions if multiword_expressions is not None else []
        self.multiword_expressions.sort(key=len, reverse=True)

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        # TODO: Add support for lower-casing and multi-word expressions
        if self.lowercase:
            input_tokens = [token.lower() for token in input_tokens]
        if self.multiword_expressions:
            processed_tokens = []
            self.multiword_expressions.sort(key=len, reverse=True)
            multiword_expressions_tokens = [token.split() for token in self.multiword_expressions]
            
            i = 0
            while i < len(input_tokens):
                matched = False
                for phrase in multiword_expressions_tokens:
                    phrase_compressed = "".join(phrase)
                    length = len(phrase_compressed)
                    i_ori = i
                    token_char_cnt = 0
                    token_i = i  # Use a separate index for token traversal
                    this_matched = True
                    for char_cnt in range(length):
                        if token_char_cnt >= len(input_tokens[token_i]):
                            token_i += 1
                            if token_i >= len(input_tokens):
                                this_matched = False
                                break
                            token_char_cnt = 0
                        if phrase_compressed[char_cnt] != input_tokens[token_i][token_char_cnt]:
                            this_matched = False
                            break
                        token_char_cnt += 1
                    if not this_matched:
                        continue  # Try the next MWE
                    last_token = input_tokens[token_i]
                    remaining = last_token[token_char_cnt:] if token_char_cnt < len(last_token) else ''
                    if remaining and not all(char in string.punctuation for char in remaining):
                        this_matched = False
                        continue  # Remaining characters are not all punctuation
                    # Match found
                    processed_tokens.append(" ".join(phrase))
                    # Append any trailing punctuation as separate tokens
                    if remaining:
                        for punct in remaining:
                            processed_tokens.append(punct)
                    # Advance i to the token after the last matched token
                    i = token_i + 1
                    matched = True
                    break  # Exit the MWE loop since we've found a match
                if not matched:
                    processed_tokens.append(input_tokens[i])
                    i += 1
            return processed_tokens
        else:
            return input_tokens
                
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # You should implement this in a subclass, not here
        raise NotImplementedError


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        tokens = text.split()
        return self.postprocess(tokens)


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        # TODO: Initialize the NLTK's RegexpTokenizer 
        self.tokenizer = RegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        tokens = self.tokenizer.tokenize(text)
        return self.postprocess(tokens)


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner", "tagger", "attribute_ruler", "lemmatizer"])
    

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single words.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        
        tokens = self.postprocess(tokens)
        return tokens


# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    pass
