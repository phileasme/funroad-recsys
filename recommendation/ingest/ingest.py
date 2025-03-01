import os
import time
from elasticsearch import Elasticsearch
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use absolute imports
from core.clip_embeddings import CLIPEmbedding
from core.colbert_embeddings import ColBERTEmbedding
from utils.gr_extract_products import get_products_data
from app import ProductDataHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Wait for Elasticsearch to be ready
    logger.info("Waiting for Elasticsearch to be ready...")
    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    es_username = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
    es_password = os.getenv("ELASTICSEARCH_PASSWORD", "changeme")
    
    for _ in range(30):  # Try for about 30 seconds
        try:
            client = Elasticsearch(es_url, basic_auth=(es_username, es_password))
            if client.ping():
                logger.info("Successfully connected to Elasticsearch")
                break
        except Exception as e:
            logger.info(f"Waiting for Elasticsearch... ({e})")
            time.sleep(1)
    else:
        raise Exception("Could not connect to Elasticsearch after multiple attempts")

    # Initialize handler with the Elasticsearch client
    logger.info("Initializing Product Data Handler...")
    handler = ProductDataHandler(client, batch_size=50)
    
    # Delete and recreate index
    logger.info("Setting up Elasticsearch index...")
    handler.delete_index()
    handler.create_index()
    
    # Initialize CLIP model for embeddings
    logger.info("Initializing CLIP embedding model...")
    try:
        # Always load model directly from HuggingFace or local directory
        model_path = "/app/recommendation/models/clip"
        os.makedirs(model_path, exist_ok=True)
        
        # Log device info
        import torch
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        clip_embedding = CLIPEmbedding()
        logger.info("CLIP model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing CLIP model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Exit with error code

    # Initialize ColBERT model for embeddings
    logger.info("Initializing ColBERT embedding model...")
    try:
        # Always load model directly from HuggingFace or local directory
        model_path = "/app/recommendation/models/colbert"
        os.makedirs(model_path, exist_ok=True)
        
        colbert_embedding = ColBERTEmbedding()
        logger.info("ColBERT model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ColBERT model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Exit with error code

    # Get products data
    logger.info("Loading product data...")
    products, keywords = get_products_data()
    
    if not products:
        logger.error("No products found! Check if gumroad_data directory contains valid data.")
        return

    # Index all products
    logger.info(f"Indexing {len(products)} products...")
    handler.index_all_products(list(products.values()))

    # Process embeddings
    logger.info("Processing product embeddings...")
    handler.process_product_embeddings(list(products.values()), clip_embedding, colbert_embedding)
    
    logger.info("Data ingestion complete!")

if __name__ == "__main__":
    main()