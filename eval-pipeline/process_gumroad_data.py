import os
import json
import glob
from datetime import datetime
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_query_from_folder(folder_name):
    """Extract the query string from a folder name."""
    if folder_name.startswith("search_"):
        query = folder_name[7:].replace("_", " ")
    else:
        query = folder_name.replace("_", " ")
    return query

def get_gumroad_results(data_dir="gumroad_data", max_queries=50):
    """
    Process all Gumroad data folders and extract search results.
    
    Args:
        data_dir: Directory containing Gumroad data
        max_queries: Maximum number of queries to process
        
    Returns:
        Dictionary mapping queries to their search results
    """
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist")
        return {}
    
    # Get all search directories
    search_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("search_")]
    
    logger.info(f"Found {len(search_dirs)} search directories")
    
    # Limit number of queries if needed
    if max_queries and len(search_dirs) > max_queries:
        search_dirs = search_dirs[:max_queries]
    
    results = {}
    
    for search_dir in search_dirs:
        query = extract_query_from_folder(search_dir)
        logger.info(f"Processing query: '{query}'")
        
        dir_path = os.path.join(data_dir, search_dir)
        
        # Find all product search files
        product_files = glob.glob(os.path.join(dir_path, "*_products_search*.json"))
        
        if not product_files:
            logger.warning(f"No product search files found for query '{query}'")
            continue
            
        # Sort files by creation time (oldest first)
        product_files.sort(key=os.path.getctime)
        
        # Collect all products from all files
        all_products = []
        
        for file_path in product_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'products' in data:
                    products = data['products']
                    logger.info(f"Found {len(products)} products in {os.path.basename(file_path)}")
                    
                    # Process each product and add rank information
                    start_rank = len(all_products) + 1
                    for i, product in enumerate(products):
                        processed_product = {
                            'gumroad_rank': start_rank + i,
                            'id': product.get('id', ''),
                            'name': product.get('name', ''),
                            'description': product.get('description', ''),
                            'thumbnail_url': product.get('thumbnail_url', ''),
                            'price_cents': product.get('price_cents', 0),
                            'ratings_count': product.get('number_of_ratings', 0),
                            'ratings_score': product.get('average_rating', 0),
                            'seller_name': product.get('seller_name', ''),
                            'seller_id': product.get('seller_id', ''),
                            'url': product.get('url', '')
                        }
                        all_products.append(processed_product)
                else:
                    logger.warning(f"No 'products' key found in {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Store results for this query
        if all_products:
            results[query] = all_products
            logger.info(f"Stored {len(all_products)} products for query '{query}'")
        else:
            logger.warning(f"No products found for query '{query}'")
    
    logger.info(f"Processed {len(results)} queries in total")
    return results

def save_gumroad_results(results, output_dir="gumroad_processed"):
    """
    Save processed Gumroad results to JSON files.
    
    Args:
        results: Dictionary mapping queries to their search results
        output_dir: Directory to save the processed results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    for query, products in results.items():
        # Create a safe filename
        safe_query = query.replace(" ", "_").replace("/", "_").replace("\\", "_")
        filename = os.path.join(output_dir, f"{safe_query}_gumroad.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'product_count': len(products),
                    'products': products
                }, f, indent=2)
            logger.info(f"Saved {len(products)} products for query '{query}' to {filename}")
        except Exception as e:
            logger.error(f"Error saving results for query '{query}': {e}")
    
    logger.info(f"All results saved to {output_dir}")

if __name__ == "__main__":
    gumroad_results = get_gumroad_results()
    save_gumroad_results(gumroad_results)
