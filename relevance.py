import csv
import math
"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""

def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    relevant_retrieved = 0
    precision_at_k = 0.0
    total_relevant = sum(search_result_relevances)
    if total_relevant == 0:
        return 0.0
    for idx, relevance in enumerate(search_result_relevances[:cut_off]):
        if relevance > 0:  
            relevant_retrieved += 1
            
            precision_at_k += relevant_retrieved / (idx + 1) 

    return precision_at_k / total_relevant if total_relevant > 0 else 0.0




def ndcg_score(search_result_relevances: list[float], ideal_relevance_score_ordering: list[float], cut_off: int = 10) -> float:
    """
    Calculate NDCG score given a list of relevance scores and the ideal ordering.
    Args:
        search_result_relevances: Relevance scores of retrieved documents.
        ideal_relevance_score_ordering: The ideal ordering of relevance scores (sorted in descending order).
        cut_off: Number of top results to consider in NDCG calculation (default 10).
    Returns:
        NDCG score (float).
    """
    def dcg(relevances):
        dcg_value = 0.0
        for i, rel in enumerate(relevances[:cut_off]):
            rank = i + 1
            dcg_value += rel / math.log2(rank + 1)
        return dcg_value
    
    dcg_score = dcg(search_result_relevances)
    idcg_score = dcg(ideal_relevance_score_ordering)
    
    if idcg_score == 0.0:
        return 0.0
    else:
        ndcg = dcg_score / idcg_score
    return ndcg


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    query_relevance = {}

    # TODO: Load the relevance dataset
    with open(relevance_data_filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            query_id = row['query_id']
            doc_id = int(row['doc_id'])
            rel = int(row['rel'])
            if query_id not in query_relevance:
                query_relevance[query_id] = {}
            query_relevance[query_id][doc_id] = rel
            
    # TODO: Run each of the dataset's queries through your ranking function

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out
    map_scores = []
    ndcg_scores = []
    for query_id, doc_relevances in query_relevance.items():
        query = query_id  # Assuming the query_id represents the query text in the dataset
        retrieved_docs = ranker.query(query)
        
        retrieved_docids = [docid for docid, score in retrieved_docs]

        relevance_scores = []
        for docid in retrieved_docids:
            rel = doc_relevances.get(docid, 0)
            relevance_scores.append(rel)
            
        ideal_relevance_ordering = sorted(doc_relevances.values(), reverse=True)
        
        search_result_relevances = [1 if rel >= 1 else 0 for rel in relevance_scores]

        map_score_value = map_score(search_result_relevances)
        ndcg_score_value = ndcg_score(search_result_relevances, ideal_relevance_ordering)
        
        map_scores.append(map_score_value)
        ndcg_scores.append(ndcg_score_value)
    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
  
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    return {'map': sum(map_scores) / len(map_scores if map_scores else 0.0), 
            'ndcg': sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0, 
            'map_list': map_scores, 
            'ndcg_list': ndcg_scores
            }


if __name__ == '__main__':
    pass
