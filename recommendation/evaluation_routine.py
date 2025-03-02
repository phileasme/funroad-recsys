

def get_query_similarity_matrix(data_dir: str = "gumroad_data", method: str = "jaccard") -> Tuple[List[str], np.ndarray]:
    """
    Compute a similarity matrix between all queries.
    
    Args:
        data_dir: Directory containing Gumroad data
        method: Similarity method - 'jaccard' for word overlap or 'embedding' for semantic similarity
        
    Returns:
        Tuple of (list of queries, similarity matrix)
    """
    # Get all search folders
    search_folders = [os.path.basename(x) for x in glob.glob(os.path.join(data_dir, "search_*"))]
    if not search_folders:
        # Also look for any other directories in case the "search_" prefix isn't used
        search_folders = [os.path.basename(x) for x in glob.glob(os.path.join(data_dir, "*")) 
                         if os.path.isdir(x)]
    
    if not search_folders:
        raise ValueError(f"No search folders found in {data_dir}")
    
    # Extract queries from folder names
    queries = [extract_query_from_folder(folder) for folder in search_folders]
    print(f"Found {len(queries)} unique queries")
    
    n = len(queries)
    similarity_matrix = np.zeros((n, n))
    
    if method == "jaccard":
        # Preprocess queries: lowercase and tokenize
        processed_queries = []
        for query in queries:
            # Convert to lowercase and split into words
            words = query.lower().split()
            processed_queries.append(set(words))
        
        # Compute Jaccard similarity (word overlap) between all pairs of queries
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Same query
                else:
                    words_i = processed_queries[i]
                    words_j = processed_queries[j]
                    
                    # Jaccard similarity: size of intersection / size of union
                    intersection = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    
                    if union > 0:
                        similarity_matrix[i, j] = intersection / union
    
    elif method == "embedding":
        try:
            # Import required modules for embeddings
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Load model for query embeddings (using a ColBERT v2 compatible model)
            print("Loading embedding model...")
            model_name = "colbert-ir/colbertv2.0"  # Can be adjusted to use other models
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Function to get embeddings for a text
            def get_embedding(text):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use CLS token embedding as query representation
                return outputs.last_hidden_state[:, 0, :].numpy()
            
            # Generate embeddings for all queries
            embeddings = [get_embedding(query) for query in queries]
            
            # Compute cosine similarity between embeddings
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        similarity_matrix[i, j] = 1.0  # Same query
                    else:
                        # Compute cosine similarity between the embeddings
                        dot_product = np.dot(embeddings[i].flatten(), embeddings[j].flatten())
                        norm_i = np.linalg.norm(embeddings[i])
                        norm_j = np.linalg.norm(embeddings[j])
                        similarity = dot_product / (norm_i * norm_j)
                        
                        # Store in both positions of the symmetric matrix
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
                        
        except ImportError:
            print("Warning: Could not import required modules for embeddings.")
            print("Falling back to Jaccard similarity.")
            # Recursively call with jaccard method
            return get_query_similarity_matrix(data_dir, "jaccard")
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return queries, similarity_matrix

def select_diverse_queries(queries: List[str], similarity_matrix: np.ndarray, num_queries: int = 40) -> List[str]:
    """
    Select a diverse set of queries with minimal word overlap.
    
    Args:
        queries: List of all available queries
        similarity_matrix: Matrix of pairwise similarity scores
        num_queries: Number of diverse queries to select
        
    Returns:
        List of selected diverse queries
    """
    if len(queries) <= num_queries:
        print(f"Only {len(queries)} queries available, returning all")
        return queries
    
    # Start with the longest query as a heuristic for richness
    query_lengths = [len(q.split()) for q in queries]
    selected_indices = [np.argmax(query_lengths)]
    remaining_indices = set(range(len(queries)))
    remaining_indices.remove(selected_indices[0])
    
    # Greedy algorithm: iteratively select the query that has minimum
    # maximum similarity to any already selected query
    while len(selected_indices) < num_queries and remaining_indices:
        min_max_similarity = float('inf')
        next_idx = -1
        
        for idx in remaining_indices:
            # Calculate maximum similarity to any already selected query
            max_sim = max(similarity_matrix[idx, selected] for selected in selected_indices)
            
            if max_sim < min_max_similarity:
                min_max_similarity = max_sim
                next_idx = idx
        
        if next_idx != -1:
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        else:
            break
    
    selected_queries = [queries[idx] for idx in selected_indices]
    
    # Sort by length for better readability
    selected_queries.sort(key=len, reverse=True)
    
    return selected_queries

def save_diverse_queries(queries: List[str], output_file: str = "diverse_queries.json"):
    """
    Save the selected diverse queries to a file.
    
    Args:
        queries: List of selected diverse queries
        output_file: Path to save the queries
    """
    with open(output_file, 'w') as f:
        json.dump({
            "count": len(queries),
            "timestamp": datetime.now().isoformat(),
            "queries": queries
        }, f, indent=2)
    
    print(f"Saved {len(queries)} diverse queries to {output_file}")

def find_diverse_query_set(data_dir: str = "gumroad_data", 
                          num_queries: int = 40, 
                          output_file: str = "diverse_queries.json",
                          similarity_method: str = "jaccard"):
    """
    Find and save a diverse set of queries with minimal similarity.
    
    Args:
        data_dir: Directory containing Gumroad data
        num_queries: Number of diverse queries to select
        output_file: Path to save the queries
        similarity_method: Method to use for computing query similarity
                          ('jaccard' for word overlap, 'embedding' for semantic similarity)
        
    Returns:
        List of selected diverse queries
    """
    print(f"Computing query similarity using method: {similarity_method}")
    queries, similarity_matrix = get_query_similarity_matrix(data_dir, similarity_method)
    diverse_queries = select_diverse_queries(queries, similarity_matrix, num_queries)
    save_diverse_queries(diverse_queries, output_file)
    
    print(f"\nSelected {len(diverse_queries)} diverse queries:")
    for i, query in enumerate(diverse_queries, 1):
        print(f"{i:2d}. {query}")
    
    return diverse_queries

import os
import json
import glob
import random
import argparse
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime
import anthropic
import pickle

client = anthropic.Anthropic(api_key="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")

def extract_query_from_folder(folder_name: str) -> str:
    """
    Extract the search query from a folder name.
    
    Args:
        folder_name: Name of the folder (e.g., 'search_3d')
        
    Returns:
        The extracted query
    """
    # Strip "search_" prefix
    if folder_name.startswith("search_"):
        query = folder_name.replace("search_", "", 1)
    else:
        query = folder_name
        
    # Replace underscores with spaces for readability
    query = query.replace("_", " ")
    
    return query

def load_gumroad_data(folder_name: str, data_dir: str = "gumroad_data") -> Tuple[str, List[Dict]]:
    """
    Load the most recent Gumroad search results for a given folder.
    
    Args:
        folder_name: Name of the folder containing search results
        data_dir: Directory containing the Gumroad data
        
    Returns:
        Tuple of (query, list of product data with Gumroad rankings)
    """
    search_dir = os.path.join(data_dir, folder_name)
    if not os.path.exists(search_dir):
        raise ValueError(f"No data found for folder '{folder_name}' in {data_dir}")
        
    # Extract query from folder name
    query = extract_query_from_folder(folder_name)
    
    # Find the most recent file
    files = glob.glob(os.path.join(search_dir, "*_products_search.json"))
    if not files:
        # Also try with "product_search.json" pattern
        files = glob.glob(os.path.join(search_dir, "*_product_search.json"))
        
    if not files:
        raise ValueError(f"No product search files found in folder '{folder_name}'")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading data from {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Extract and format product information
    formatted_data = []
    
    # Handle different possible JSON structures
    products_list = data.get("products", [])
    if not products_list and isinstance(data, list):
        # In case the JSON is directly a list of products
        products_list = data
    
    for rank, product in enumerate(products_list, 1):
        formatted_data.append({
            "gumroadrank": rank,
            "productid": product.get("id", ""),
            "name": product.get("name", ""),
            "description": product.get("description", ""),
            "ratingscore": product.get("average_rating", 0),
            "numberofratings": product.get("number_of_ratings", 0)
        })
    
    return query, formatted_data

def create_dataset(query: str, products: List[Dict], output_file: str = None) -> Dict:
    """
    Create and save a dataset associating the query with the products.
    
    Args:
        query: The search query
        products: List of product data with Gumroad rankings
        output_file: Optional file to save the dataset
        
    Returns:
        Dataset dictionary
    """
    dataset = {
        "query": query,
        "products": products
    }
    
    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {output_file}")
    
    return dataset

def get_llm_ranking(query: str, products: List[Dict]) -> List[Dict]:
    """
    Use LLM to rank products based on relevance to the query.
    
    Args:
        query: The search query
        products: List of product data (without Gumroad ranking)
        
    Returns:
        Products with added LLM ranking
    """
    # Prepare randomized product list without Gumroad ranking
    shuffled_products = [{k: v for k, v in p.items() if k != 'gumroadrank'} 
                         for p in products]
    random.shuffle(shuffled_products)
    
    # Construct prompt for the LLM
    prompt = f"""Rank the following products based on their relevance to the search query: "{query}"

Your task is to order these products from most relevant (rank 1) to least relevant, considering:
1. How well the name matches the query
2. How well the description matches the query
3. The quality indicated by rating score and number of ratings

Here are the products (in no particular order):

{json.dumps(shuffled_products, indent=2)}

Provide output as a JSON array of product IDs in order from most relevant to least relevant:
[productid1, productid2, ...]
"""

    # Call Claude API
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="You are a helpful assistant that ranks products based on relevance to a query. Output only valid JSON.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse the response to get ordered product IDs
    try:
        # Extract just the JSON part from the response
        content = response.content[0].text
        # Find JSON array in the content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            ordered_product_ids = json.loads(json_str)
        else:
            raise ValueError("Could not find JSON array in response")
            
        # Map back to get full product data with LLM rank
        product_map = {p["productid"]: p for p in products}
        ranked_products = []
        
        for llm_rank, product_id in enumerate(ordered_product_ids, 1):
            if product_id in product_map:
                product = product_map[product_id].copy()
                product["llmrank"] = llm_rank
                ranked_products.append(product)
            else:
                print(f"Warning: Product ID {product_id} not found in original data")
        
        # Add any missing products at the end
        missing_products = [p for p in products if p["productid"] not in ordered_product_ids]
        for i, product in enumerate(missing_products, len(ranked_products) + 1):
            product_copy = product.copy()
            product_copy["llmrank"] = i
            ranked_products.append(product_copy)
            
        return ranked_products
        
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response.content[0].text}")
        # Return original products without LLM ranking on error
        return [dict(p, llmrank=0) for p in products]

def compute_metrics(products: List[Dict], k_values: List[int] = [5, 10, 20]) -> Dict:
    """
    Compute ranking metrics: Precision@k, Recall@k, NDCG@k, and MRR.
    
    Args:
        products: List of product data with both Gumroad and LLM rankings
        k_values: List of k values for the metrics
        
    Returns:
        Dictionary of metrics
    """
    # Sort products by Gumroad rank (ground truth)
    gumroad_sorted = sorted(products, key=lambda x: x["gumroadrank"])
    gumroad_ids = [p["productid"] for p in gumroad_sorted]
    
    # Sort products by LLM rank
    llm_sorted = sorted(products, key=lambda x: x["llmrank"])
    llm_ids = [p["productid"] for p in llm_sorted]
    
    metrics = {}
    
    # Calculate Precision@k and Recall@k
    precision_at_k = {}
    recall_at_k = {}
    for k in k_values:
        if k > len(gumroad_ids):
            continue
        # Define relevant items as top-k in Gumroad results
        relevant = set(gumroad_ids[:k])
        retrieved = set(llm_ids[:k])
        
        # Precision: proportion of retrieved items that are relevant
        precision = len(relevant.intersection(retrieved)) / k
        precision_at_k[f"precision@{k}"] = precision
        
        # Recall: proportion of relevant items that are retrieved
        recall = len(relevant.intersection(retrieved)) / len(relevant) if relevant else 0
        recall_at_k[f"recall@{k}"] = recall
    
    metrics["precision"] = precision_at_k
    metrics["recall"] = recall_at_k
    
    # Calculate NDCG@k
    ndcg_at_k = {}
    for k in k_values:
        if k > len(gumroad_ids):
            continue
        
        # Create relevance dictionary (higher ranks have higher relevance)
        max_rank = len(gumroad_ids)
        relevance = {pid: max_rank - idx for idx, pid in enumerate(gumroad_ids)}
        
        # Calculate DCG for LLM ranking
        dcg = 0
        for i, pid in enumerate(llm_ids[:k], 1):
            rel = relevance.get(pid, 0)
            dcg += rel / np.log2(i + 1)
        
        # Calculate ideal DCG
        idcg = 0
        for i in range(1, k + 1):
            rel = max_rank - (i - 1)
            idcg += rel / np.log2(i + 1)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_at_k[f"ndcg@{k}"] = ndcg
    
    metrics["ndcg"] = ndcg_at_k
    
    # Calculate Mean Reciprocal Rank (MRR)
    mrr = 0
    for i, pid in enumerate(gumroad_ids[:1]):  # Only consider the top Gumroad result
        if pid in llm_ids:
            llm_rank = llm_ids.index(pid) + 1
            mrr = 1 / llm_rank
            break
    
    metrics["mrr"] = mrr
    
    return metrics

def run_evaluation(folder_name: str, data_dir: str = "gumroad_data", output_dir: str = "evaluation_results"):
    """
    Run the complete evaluation process.
    
    Args:
        folder_name: Name of the folder containing search results
        data_dir: Directory containing the Gumroad data
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and format Gumroad data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query, products = load_gumroad_data(folder_name, data_dir)
    dataset_file = os.path.join(output_dir, f"{folder_name}_{timestamp}_dataset.pkl")
    dataset = create_dataset(query, products, dataset_file)
    
    # Step 2: Get LLM ranking
    ranked_products = get_llm_ranking(query, products)
    
    # Save the complete ranking results
    results_file = os.path.join(output_dir, f"{query}_{timestamp}_ranking_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "query": query,
            "ranked_products": ranked_products
        }, f, indent=2)
    print(f"Ranking results saved to {results_file}")
    
    # Step 3: Compute evaluation metrics
    metrics = compute_metrics(ranked_products)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f"{query}_{timestamp}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "query": query,
            "metrics": metrics
        }, f, indent=2)
    print(f"Evaluation metrics saved to {metrics_file}")
    
    # Print metrics summary
    print("\nEvaluation Metrics Summary:")
    print(f"Query: {query}")
    print("Precision@k:")
    for k, val in metrics["precision"].items():
        print(f"  {k}: {val:.4f}")
    print("Recall@k:")
    for k, val in metrics["recall"].items():
        print(f"  {k}: {val:.4f}")
    print("NDCG@k:")
    for k, val in metrics["ndcg"].items():
        print(f"  {k}: {val:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    
    return {
        "dataset": dataset,
        "ranked_products": ranked_products,
        "metrics": metrics
    }

def compare_rankings(custom_ranked_products: List[Dict], gumroad_ranked_products: List[Dict], 
                    k_values: List[int] = [5, 10, 20]) -> Dict:
    """
    Compare a custom ranking against Gumroad's ranking.
    
    Args:
        custom_ranked_products: Products ranked by a custom algorithm
        gumroad_ranked_products: Products ranked by Gumroad
        k_values: List of k values for the metrics
        
    Returns:
        Dictionary of metrics
    """
    # Ensure both lists have the same products
    product_map = {p["productid"]: p for p in gumroad_ranked_products}
    
    # Add Gumroad rankings to custom ranked products if needed
    for product in custom_ranked_products:
        pid = product["productid"]
        if pid in product_map and "gumroadrank" not in product:
            product["gumroadrank"] = product_map[pid]["gumroadrank"]
    
    # Compute metrics
    return compute_metrics(custom_ranked_products, k_values)

def evaluate_all_folders(data_dir: str = "gumroad_data", output_dir: str = "evaluation_results"):
    """
    Evaluate all search folders in the data directory.
    
    Args:
        data_dir: Directory containing Gumroad data
        output_dir: Directory to save results
    """
    search_folders = [os.path.basename(x) for x in glob.glob(os.path.join(data_dir, "search_*"))]
    if not search_folders:
        # Also look for any other directories in case the "search_" prefix isn't used
        search_folders = [os.path.basename(x) for x in glob.glob(os.path.join(data_dir, "*")) 
                         if os.path.isdir(x)]
    
    if not search_folders:
        print(f"No search folders found in {data_dir}")
        return
    
    results = {}
    for folder in search_folders:
        print(f"\nProcessing folder: {folder}")
        try:
            result = run_evaluation(folder, data_dir, output_dir)
            results[folder] = result
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
    
    # Compile summary report
    summary_file = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    summary = {
        "folders_processed": len(results),
        "timestamp": datetime.now().isoformat(),
        "folder_metrics": {
            folder: {
                "query": extract_query_from_folder(folder),
                "metrics": result["metrics"]
            } for folder, result in results.items()
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary report saved to {summary_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate product ranking quality")
    parser.add_argument("--folder", help="Specific search folder to evaluate (optional)")
    parser.add_argument("--data-dir", default="gumroad_data", help="Directory containing Gumroad data")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory to save results")
    parser.add_argument("--find-diverse-queries", action="store_true", 
                        help="Find diverse queries with minimal overlap")
    parser.add_argument("--num-diverse", type=int, default=40,
                        help="Number of diverse queries to select")
    parser.add_argument("--similarity-method", choices=["jaccard", "embedding"], default="jaccard",
                        help="Method to use for computing query similarity")
    parser.add_argument("--evaluate-diverse-only", action="store_true",
                        help="Only evaluate the diverse query set")
    args = parser.parse_args()
    
    if args.find_diverse_queries:
        diverse_queries = find_diverse_query_set(args.data_dir, args.num_diverse, 
                                                similarity_method=args.similarity_method)
        
        if args.evaluate_diverse_only:
            # Evaluate only the diverse queries
            results = {}
            for query in diverse_queries:
                # Convert query back to folder name format
                folder = "search_" + query.replace(" ", "_")
                print(f"\nProcessing query: {query} (folder: {folder})")
                try:
                    result = run_evaluation(folder, args.data_dir, args.output_dir)
                    results[query] = result
                except Exception as e:
                    print(f"Error processing query '{query}': {e}")
    elif args.folder:
        run_evaluation(args.folder, args.data_dir, args.output_dir)
    else:
        evaluate_all_folders(args.data_dir, args.output_dir)