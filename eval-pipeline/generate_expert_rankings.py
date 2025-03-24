import os
import json
import time
import logging
import anthropic
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Anthropic client
client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here"))

def rank_results_with_llm(query, products, model="claude-3-opus-20240229", max_products=10):
    """
    Use Anthropic's Claude to rank search results by relevance.
    
    Args:
        query: Search query
        products: List of product data
        model: Anthropic model to use
        max_products: Maximum number of products to rank
        
    Returns:
        Ranked list of product IDs
    """
    # Limit the number of products to avoid token limits
    # Note: products should already be shuffled before calling this function
    products_to_rank = products[:max_products]
    
    # Prepare product list for the prompt
    products_text = []
    for i, product in enumerate(products_to_rank):
        # Extract the most relevant fields for ranking
        product_text = f"Product {i+1}:\n"
        product_text += f"- ID: {product.get('id', 'unknown')}\n"
        product_text += f"- Name: {product.get('name', 'Untitled')}\n"
        product_text += f"- Description: {product.get('description', 'No description')[:500]}{'...' if len(product.get('description', '')) > 500 else ''}\n"
        product_text += f"- Price: ${product.get('price_cents', 0) / 100:.2f}\n"
        
        ratings_count = product.get('ratings_count', 0)
        ratings_score = product.get('ratings_score', 0)
        if ratings_count > 0:
            product_text += f"- Ratings: {ratings_score:.1f}/5 ({ratings_count} reviews)\n"
        else:
            product_text += "- Ratings: None\n"
            
        product_text += f"- Seller: {product.get('seller_name', 'Unknown')}\n"
        products_text.append(product_text)
    
    products_str = "\n".join(products_text)
    
    # Construct the prompt
    prompt = f"""You are an expert search engine evaluator. Your task is to rank the following products based on their relevance to the search query: "{query}"

When determining relevance, please consider:
1. How well the product name and description match the search intent
2. Overall quality (based on ratings, when available)
3. Comprehensiveness and clarity of product information

Here are the products to rank:

{products_str}

Please provide your ranking as a JSON list of product IDs in order from most relevant to least relevant:
["{products_to_rank[0].get('id', 'unknown')}", "{products_to_rank[1].get('id', 'unknown')}", ...]

First think step by step about what would make a product relevant to this query, then evaluate each product in detail before finalizing your ranking. Do not skip or omit any products from your ranking.
"""

    try:
        # Send request to Anthropic
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,  # Use deterministic output
            system="You are an expert search engine evaluator that ranks products based on their relevance to search queries. Respond only with valid JSON.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the JSON from the response
        content = response.content[0].text
        
        # Find the JSON array in the content
        import re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        
        if json_match:
            try:
                ranked_ids = json.loads(json_match.group(0))
                logger.info(f"Successfully ranked {len(ranked_ids)} products for query '{query}'")
                return ranked_ids
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {e}")
                # Try a more lenient extraction
                array_text = content[content.find('['):content.rfind(']')+1]
                try:
                    # Clean up the text by removing newlines and ensuring proper quotes
                    clean_text = array_text.replace('\n', '').replace("'", '"')
                    ranked_ids = json.loads(clean_text)
                    logger.info(f"Successfully parsed JSON after cleanup for query '{query}'")
                    return ranked_ids
                except:
                    logger.error(f"Failed to parse JSON even after cleanup")
                    return None
        else:
            logger.error("Could not find JSON array in response")
            logger.debug(f"Response content: {content}")
            return None
        
    except Exception as e:
        logger.error(f"Error in Anthropic API call: {e}")
        return None

def process_gumroad_endpoint_pairs(input_dir_gumroad, input_dir_endpoint, output_dir, max_workers=1):
    """
    Process pairs of Gumroad and endpoint results to get expert rankings.
    
    Args:
        input_dir_gumroad: Directory containing Gumroad results
        input_dir_endpoint: Directory containing endpoint results
        output_dir: Directory to save expert rankings
        max_workers: Maximum number of concurrent workers
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Get all Gumroad JSON files
    gumroad_files = [f for f in os.listdir(input_dir_gumroad) if f.endswith('_gumroad.json')]
    logger.info(f"Found {len(gumroad_files)} Gumroad result files")
    
    # Define worker function for parallel processing
    def process_query_pair(gumroad_file):
        try:
            # Extract query from filename
            query_name = gumroad_file.replace('_gumroad.json', '')
            endpoint_file = f"{query_name}_endpoint.json"
            
            # Check if endpoint file exists
            if not os.path.exists(os.path.join(input_dir_endpoint, endpoint_file)):
                logger.warning(f"No endpoint file found for {query_name}")
                return False
            
            # Load Gumroad results
            with open(os.path.join(input_dir_gumroad, gumroad_file), 'r', encoding='utf-8') as f:
                gumroad_data = json.load(f)
                query = gumroad_data.get('query', '')
                gumroad_products = gumroad_data.get('products', [])
            
            # Load endpoint results
            with open(os.path.join(input_dir_endpoint, endpoint_file), 'r', encoding='utf-8') as f:
                endpoint_data = json.load(f)
                endpoint_results = endpoint_data.get('results', {}).get('results', [])
            
            if not query or not gumroad_products or not endpoint_results:
                logger.warning(f"Missing data for {query_name}")
                return False
            
            logger.info(f"Processing query: '{query}' with {len(gumroad_products)} Gumroad products and {len(endpoint_results)} endpoint results")
            
            # Create combined product list for ranking (include top results from both)
            combined_products = []
            seen_ids = set()
            
            # Add Gumroad products
            for product in gumroad_products[:20]:  # Limit to top 20
                product_id = product.get('id', '')
                if product_id and product_id not in seen_ids:
                    seen_ids.add(product_id)
                    # Add source metadata but don't expose it in the prompt
                    product_copy = product.copy()
                    product_copy['_source'] = 'gumroad'
                    combined_products.append(product_copy)
            
            # Add endpoint products
            for result in endpoint_results[:20]:  # Limit to top 20
                product_id = result.get('id', '')
                if product_id and product_id not in seen_ids:
                    seen_ids.add(product_id)
                    
                    # Convert endpoint result to same format as Gumroad products
                    endpoint_product = {
                        'id': product_id,
                        'name': result.get('name', ''),
                        'description': result.get('description', ''),
                        'thumbnail_url': result.get('thumbnail_url', ''),
                        'price_cents': result.get('price_cents', 0),
                        'ratings_count': result.get('ratings_count', 0),
                        'ratings_score': result.get('ratings_score', 0),
                        'seller_name': result.get('seller_name', ''),
                        '_source': 'endpoint'  # Add source metadata but don't expose it in the prompt
                    }
                    combined_products.append(endpoint_product)
                    
            # Randomize the order of products to avoid positional bias
            import random
            random.shuffle(combined_products)
            
            # Get expert ranking
            ranked_product_ids = rank_results_with_llm(query, combined_products)
            
            if not ranked_product_ids:
                logger.warning(f"Failed to get expert ranking for '{query}'")
                return False
            
            # Save expert ranking
            output_file = os.path.join(output_dir, f"{query_name}_expert.json")
            
            # Find product info for each ranked ID
            ranked_products = []
            id_to_product = {p.get('id', ''): p for p in combined_products}
            
            for product_id in ranked_product_ids:
                if product_id in id_to_product:
                    ranked_products.append(id_to_product[product_id])
            
            # Record source statistics for later analysis
            ranked_sources = []
            for product_id in ranked_product_ids:
                for product in combined_products:
                    if product.get('id') == product_id:
                        source = product.get('_source', 'unknown')
                        ranked_sources.append(source)
                        break
            
            source_stats = {
                'gumroad_count': ranked_sources.count('gumroad'),
                'endpoint_count': ranked_sources.count('endpoint'),
                'unknown_count': ranked_sources.count('unknown'),
                'position_analysis': [
                    {'position': i+1, 'source': source} 
                    for i, source in enumerate(ranked_sources)
                ]
            }
            
            # Save expert ranking with product details
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'total_products_considered': len(combined_products),
                    'ranked_product_ids': ranked_product_ids,
                    'ranked_products': ranked_products,
                    'source_statistics': source_stats
                }, f, indent=2)
            
            logger.info(f"Saved expert ranking for '{query}' to {output_file}")
            
            # Add some delay to avoid rate limits
            time.sleep(5)
            return True
            
        except Exception as e:
            logger.error(f"Error processing {gumroad_file}: {e}")
            return False
    
    # Process files with parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_query_pair, gumroad_files))
    
    success_count = sum(results)
    logger.info(f"Processed {len(gumroad_files)} queries: {success_count} successful, {len(gumroad_files) - success_count} failed")

if __name__ == "__main__":
    # Configuration
    gumroad_processed_dir = "gumroad_processed"
    endpoint_results_dir = "endpoint_results"
    expert_rankings_dir = "expert_rankings"
    
    # Process query pairs
    process_gumroad_endpoint_pairs(gumroad_processed_dir, endpoint_results_dir, expert_rankings_dir)
