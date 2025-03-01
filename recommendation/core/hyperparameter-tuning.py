import itertools
import json
import numpy as np
import datetime

class SearchRelevanceTuner:
    # Taking a gridsearch approach, but we would take an ltr 
    # or optuna approach in a more serious environment
    def __init__(self, es_client, query_set):
        """
        Initialize tuner with Elasticsearch client and test queries
        
        :param es_client: Elasticsearch client
        :param query_set: List of test queries with expected relevant results
        """
        self.es_client = es_client
        self.query_set = query_set
    
    def compute_metrics(self, results, ground_truth):
        """
        Compute relevance metrics
        
        :param results: Retrieved search results
        :param ground_truth: Expected relevant results
        :return: Metrics dictionary
        """
        # Precision at K
        def precision_at_k(retrieved, relevant_k, k=10):
            return len(set(retrieved[:k]) & set(relevant_k)) / k
        
        # Mean Average Precision
        def mean_average_precision(retrieved, relevant):
            average_precisions = []
            for k in range(1, len(retrieved) + 1):
                precision_k = precision_at_k(retrieved, relevant, k)
                average_precisions.append(precision_k)
            return np.mean(average_precisions)
        
        return {
            'precision_at_10': precision_at_k(results, ground_truth),
            'mean_average_precision': mean_average_precision(results, ground_truth)
        }
    
    def grid_search_weights(self, weight_ranges):
        """
        Perform grid search over possible weight combinations
        
        :param weight_ranges: Dictionary of parameter ranges to search
        :return: Best weights and their performance
        """
        best_score = -np.inf
        best_weights = None
        
        # Generate all weight combinations
        param_names = list(weight_ranges.keys())
        param_values = [weight_ranges[name] for name in param_names]
        
        # Iterate through all combinations
        for combination in itertools.product(*param_values):
            weights = dict(zip(param_names, combination))
            
            # Aggregate performance across all test queries
            total_scores = []
            
            for query_data in self.query_set:
                query = query_data['query']
                ground_truth = query_data['ground_truth']
                
                # Perform search with current weights
                results = self._search_with_weights(query, weights)
                
                # Compute metrics
                metrics = self.compute_metrics(results, ground_truth)
                total_scores.append(metrics['mean_average_precision'])
            
            # Compute average performance
            avg_score = np.mean(total_scores)
            
            # Update best weights
            if avg_score > best_score:
                best_score = avg_score
                best_weights = weights
        
        return best_weights, best_score
    
    def _search_with_weights(self, query, weights):
        """
        Perform search with specific weights
        
        :param query: Search query
        :param weights: Dictionary of weights to apply
        :return: List of retrieved result IDs
        """
        # Implement your specific search logic here
        # This is a placeholder - you'd replace with your actual search method
        search_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["name^{}".format(weights.get('name_boost', 1)), 
                                           "description^{}".format(weights.get('description_boost', 1))]
                            }
                        },
                        # Add vector search with weights
                        {
                            "function_score": {
                                "query": {"match_all": {}},
                                "functions": [
                                    {
                                        "script_score": {
                                            "script": {
                                                "source": "_score * params.vector_weight",
                                                "params": {"vector_weight": weights.get('vector_weight', 1)}
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        }
        
        # Perform search and extract result IDs
        results = self.es_client.search(index='products', body=search_query)
        return [hit['_id'] for hit in results['hits']['hits']]
    
    def save_results(self, best_weights, best_score):
        """
        Save tuning results to a file
        
        :param best_weights: Best discovered weights
        :param best_score: Performance score of best weights
        """
        with open('search_weights_tuning.json', 'w') as f:
            json.dump({
                'best_weights': best_weights,
                'best_score': best_score,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

# Example usage
def tune_search_relevance(es_client):
    # Prepare test query set with ground truth
    query_set = [
        {
            'query': 'poker',
            'ground_truth': ['product_id_1', 'product_id_2']  # Known relevant product IDs
        },
        {
            'query': 'digital art',
            'ground_truth': ['product_id_3', 'product_id_4']
        }
        # Add more test queries
    ]
    
    tuner = SearchRelevanceTuner(es_client, query_set)
    
    # Define weight ranges to search
    weight_ranges = {
        'name_boost': [1, 2, 3],
        'description_boost': [1, 1.5, 2],
        'vector_weight': [0.1, 0.5, 1]
    }
    
    # Perform grid search
    best_weights, best_score = tuner.grid_search_weights(weight_ranges)
    
    # Save and log results
    tuner.save_results(best_weights, best_score)
    
    return best_weights, best_score
