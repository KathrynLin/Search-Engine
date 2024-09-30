"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from indexing import InvertedIndex
from collections import Counter
import math
from math import log

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]=None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query
        query_tokens = self.tokenize(query)
        if self.stopwords:
            query_tokens = [token for token in query_tokens if token.lower() not in self.stopwords]
        #print(f"Query Tokens: {query_tokens}")
        query_word_counts = Counter(query_tokens)

        # 2. Fetch a list of possible documents from the index
        candidate_docs = set()
        for token in query_word_counts:
            postings = self.index.get_postings(token)  
            
            for docid, term_frequency, *positions in postings:  
                candidate_docs.add(docid)
                #print(docid, term_frequency, positions)
        if not candidate_docs:
            return []
        
        # 2. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        doc_scores = []
        for docid in candidate_docs:
            #doc_word_counts = self.index.get_doc_word_counts(docid)            
            #print("doc_word_counts_1111", doc_word_counts)
            doc_word_counts = self.get_doc_word_counts(docid) 
            print("doc_word_counts", doc_word_counts)
            score = self.scorer.score(docid, doc_word_counts, query_word_counts)
            doc_scores.append((docid, score))
            
        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores

    def get_doc_word_counts(self, docid: int) -> dict[str, int]:
        """
        Retrieves the word counts for a specific document by looking up all terms in the index and 
        their respective frequencies in the document.
        """
        doc_word_counts = {}
        for term in self.index.vocabulary:  # Iterate over all terms in the vocabulary
            postings = self.index.get_postings(term)
            for doc, term_frequency, *positions in postings:
                if doc == docid:
                    doc_word_counts[term] = term_frequency
        return doc_word_counts


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        dot_product = 0.0
        # doc_word_counts = {term: count for term, count in doc_word_counts.items()}
        # query_word_counts = {term: count for term, count in query_word_counts.items()}
        
        # for term, query_count in query_word_counts.items():
        #     print("doc_word_counts", doc_word_counts)
        #     doc_count = doc_word_counts.get(term, 0)
        #     print(f"Scoring Doc {docid}: Term: {term}, Query Count: {query_count}, Doc Count: {doc_count}")

        #     dot_product += query_count * doc_count
        #     #print(f"Document Term: {term}, Document Count: {doc_count}, Dot Product: {dot_product}")
        
        for term, query_count in query_word_counts.items():
            doc_count = doc_word_counts.get(term, 0)
            print(f"Scoring Doc {docid}: Term: {term}, Query Count: {query_count}, Doc Count: {doc_count}")
            dot_product += query_count * doc_count
        if dot_product == 0:
            return 0.0
        # 2. Return the score
        # doc_magnitude = math.sqrt(sum(count ** 2 for count in doc_word_counts.values()))
        # query_magnitude = math.sqrt(sum(count ** 2 for count in query_word_counts.values()))
        # if doc_magnitude == 0 or query_magnitude == 0:
        #     return 0.0
        return dot_product

# TODO Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        return NotImplementedError


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score    

        # 4. Return score
        return NotImplementedError


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        return NotImplementedError


# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return the score
        return NotImplementedError


# TODO Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    pass
