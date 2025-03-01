import os, json
from tqdm import tqdm
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a relative path that will work within the Docker container
base_dir = "/app/gumroad_data"

def get_products_data():
    products = {}
    keywords = set()
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        logger.error(f"Directory {base_dir} does not exist!")
        return {}, set()
    
    # List the directories and files in the base_dir
    logger.info(f"Scanning {base_dir} for product data...")
    logger.info(f"Contents of base_dir: {os.listdir(base_dir)}")
    
    # Process all search directories for new keywords
    for search_dir in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, search_dir)
        if os.path.isdir(dir_path):
            logger.info(f"Processing directory: {search_dir}")
            
            for filename in os.listdir(dir_path):
                if 'products_search' in filename:
                    logger.info(f"Found product search file: {filename}")
                    try:
                        with open(os.path.join(dir_path, filename)) as f:
                            data = json.load(f)
                            
                            if 'tags_data' in data:
                                for tag in data['tags_data']:
                                    keywords.add(tag['key'].lower())
                            
                            if 'products' in data:
                                product_count = len(data['products'])
                                logger.info(f"Found {product_count} products in {filename}")
                                
                                for prod in data['products']:
                                    try:
                                        prod_reorganized = {}
                                        search_tag = "_".join(search_dir.split("_")[1:])
                                        
                                        if "search_tag" in prod:
                                            if isinstance(prod["search_tag"], set):
                                                prod["search_tag"].add(search_tag)
                                            else:
                                                prod["search_tag"] = {search_tag}
                                        else:
                                            prod["search_tag"] = {search_tag}
                                        
                                        # Extract required fields
                                        prod_reorganized["id"] = prod["id"]
                                        prod_reorganized["name"] = prod["name"]
                                        prod_reorganized["description"] = prod.get("description", "")
                                        prod_reorganized["seller"] = prod.get('seller', {})
                                        prod_reorganized["ratings"] = prod.get('ratings', {"average": 0, "count": 0})
                                        prod_reorganized["price_cents"] = prod.get("price_cents", 0)
                                        prod_reorganized["native_type"] = prod.get("native_type", "")
                                        prod_reorganized["thumbnail_url"] = prod.get("thumbnail_url", "")
                                        prod_reorganized["url"] = prod.get("url", "")
                                        
                                        products[prod["id"]] = prod_reorganized
                                    except Exception as e:
                                        logger.error(f"Error processing product: {e}")
                    except Exception as e:
                        logger.error(f"Error processing file {filename}: {e}")
    
    logger.info(f"Processed {len(products)} unique products and {len(keywords)} keywords")
    return products, keywords