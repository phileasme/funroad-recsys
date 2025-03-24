import os
import json
import requests
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_endpoint(endpoint_url, query, k=50, num_candidates=100, max_retries=3, timeout=30):
    """
    Query the search endpoint and return results.
    
    Args:
        endpoint_url: URL of the search endpoint
        query: Search query
        k: Number of results to return
        num_candidates: Number of candidates to consider
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds
        
    Returns:
        Search results or None if the request failed
    """
    payload = {
        "query": query,
        "k": k,
        "num_candidates": num_candidates
    }
    
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(endpoint_url, json=payload, headers=headers, timeout=timeout)
            query_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                logger.info(f"Query: '{query}' - Got {len(result_data.get('results', []))} results in {query_time:.2f}s")
                
                # Add query time to the response data
                if isinstance(result_data, dict):
                    result_data['query_time_seconds'] = query_time
                
                return result_data
            else:
                logger.warning(f"Query: '{query}' - Error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Query: '{query}' - Exception: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(1)  # Wait before retrying
    
    logger.error(f"Query: '{query}' - Failed after {max_retries} attempts")
    return None

def process_queries_from_file(input_dir, endpoint_url, output_dir, max_workers=5):
    """
    Process queries from Gumroad processed files and query the endpoint.
    
    Args:
        input_dir: Directory containing processed Gumroad results
        endpoint_url: URL of the search endpoint
        output_dir: Directory to save endpoint results
        max_workers: Maximum number of concurrent workers
    """
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Get all JSON files in the input directory
    gumroad_files = [f for f in os.listdir(input_dir) if f.endswith('_gumroad.json')]
    logger.info(f"Found {len(gumroad_files)} Gumroad result files to process")
    
    queries = []
    
    # Extract queries from files
    for filename in gumroad_files:
        try:
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                queries.append(data.get('query', ''))
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
    
    # Filter out empty queries
    queries = [q for q in queries if q]
    logger.info(f"Extracted {len(queries)} unique queries")
    
    # Define the worker function for parallel processing
    def process_query(query):
        logger.info(f"Processing query: '{query}'")
        
        # Query the endpoint
        endpoint_results = query_endpoint(endpoint_url, query)
        
        if endpoint_results:
            # Save the results
            safe_query = query.replace(" ", "_").replace("/", "_").replace("\\", "_")
            filename = os.path.join(output_dir, f"{safe_query}_endpoint.json")
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump({
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'endpoint_url': endpoint_url,
                        'results': endpoint_results
                    }, f, indent=2)
                logger.info(f"Saved endpoint results for query '{query}' to {filename}")
                return True
            except Exception as e:
                logger.error(f"Error saving results for query '{query}': {e}")
                return False
        
        logger.warning(f"No endpoint results for query '{query}'")
        return False
    
    # Process queries in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_query, queries))
    
    success_count = sum(results)
    logger.info(f"Processed {len(queries)} queries: {success_count} successful, {len(queries) - success_count} failed")

if __name__ == "__main__":
    # Configuration
    gumroad_processed_dir = "gumroad_processed"
    endpoint_results_dir = "endpoint_results"
    endpoint_url = "http://localhost:8000/two_phase_optimized"  # Update with your actual endpoint
    
    # Process queries
    process_queries_from_file(gumroad_processed_dir, endpoint_url, endpoint_results_dir)
