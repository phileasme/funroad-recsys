from PIL import Image
import torch
import numpy as np
from typing import List, Dict
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_downloader import process_image_url
import time
import logging

class ProductDataHandler:
    def __init__(self, es_client, batch_size=100):
        self.es = es_client
        self.batch_size = batch_size
        self.index_name = "products"  # Make sure this matches your index name
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        

    def delete_index(self):
            """
            Deletes the index if it exists
            Prints success/failure message
            """
            try:
                if self.es.indices.exists(index=self.index_name):
                    self.es.indices.delete(index=self.index_name)
                    self.logger.info(f"Successfully deleted index: {self.index_name}")
                else:
                    self.logger.info(f"Index {self.index_name} does not exist")
                    
            except Exception as e:
                self.logger.info(f"Error deleting index {self.index_name}: {e}")
                raise e

    def create_index(self):
            index_mapping = {
                "settings": {
                    "index": {
                        "refresh_interval": "5s",
                        "number_of_shards": 1,    
                        "number_of_replicas": 1 
                    },
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "analysis": {
                        "analyzer": {
                            "product_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "asciifolding",
                                    "word_delimiter_graph"
                                ]
                            },
                            "product_ngram_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "asciifolding",
                                    "edge_ngram_filter"
                                ]
                            }
                        },
                        "filter": {
                            "edge_ngram_filter": {
                                "type": "edge_ngram",
                                "min_gram": 2,
                                "max_gram": 20
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "name": {
                        "type": "text",
                        "analyzer": "product_analyzer",
                        "index_options": "positions",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "ngram": {
                            "type": "text",
                            "analyzer": "product_ngram_analyzer"
                            }
                        }
                        },
                        "description": {
                            "type": "text",
                            "index_options": "positions",
                            "analyzer": "product_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "suggest": {
                                    "type": "completion"
                                }
                            }
                        },
                        "combined_text": {
                            "type": "text",
                            "analyzer": "product_analyzer",
                            "index_options": "positions"
                        },
                        "description_embedding": {
                            "type": "dense_vector",
                            "dims": 768,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw"
                            }
                        },
                        "description_clip_embedding": {
                            "type": "dense_vector",
                            "dims": 512,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw"
                            }
                        },
                        "image_embedding": {
                            "type": "dense_vector",
                            "dims": 512,
                            "similarity": "cosine",
                            "index": True,
                            "index_options": {
                                "type": "hnsw"
                            }
                        },
                        "seller": {
                            "properties": {
                                "id": {"type": "keyword"},
                                "name": {
                                    "type": "text",
                                    "analyzer": "product_analyzer",
                                    "fields": {
                                        "keyword": {"type": "keyword"}
                                    }
                                },
                                "avatar_url": {"type": "keyword"},
                                "profile_url": {"type": "keyword"}
                            }
                        },
                        "ratings": {
                            "properties": {
                                "average": {"type": "float"},
                                "count": {"type": "integer"}
                            }
                        },
                        "price_cents": {"type": "integer"},
                        "native_type": {"type": "keyword"},
                        "thumbnail_url": {"type": "keyword"},
                        "url": {"type": "keyword"}
                    }
                }
            }

            try:
                if not self.es.indices.exists(index=self.index_name):
                    self.es.indices.create(index=self.index_name, body=index_mapping)
                    print(f"Successfully created index: {self.index_name}")
                    # Update settings using keyword arguments
                else:
                    print(f"Index {self.index_name} already exists")
            except Exception as e:
                print(f"Error creating index: {e}")
                raise e
        

    def index_all_products(self, products):
        """
        Index products in Elasticsearch using batches
        
        Args:
            products: List of product dictionaries to index
        """
        print(f"Processing {len(products)} products in batches of {self.batch_size}")
        
        for i in range(0, len(products), self.batch_size):
            batch = products[i:i + self.batch_size]
            try:
                # Create bulk actions with proper format
                actions = []
                for product in batch:
                    # Add action
                    actions.append({
                        "index": {
                            "_index": self.index_name,
                            "_id": product['id']
                        }
                    })
                    # Add document
                    actions.append(product)
                
                # Execute bulk operation
                results = self.es.bulk(operations=actions, refresh=True)
                
                # Process results
                success = sum(1 for item in results['items'] if item['index']['status'] in [200, 201])
                failed = len(results['items']) - success
                
                print(f"Batch {i//self.batch_size + 1}: Indexed {success} products, {failed} failed")
                
                # Print detailed errors if any failures
                if failed > 0:
                    for item in results['items']:
                        if item['index']['status'] not in [200, 201]:
                            print(f"Error indexing document {item['index'].get('_id')}: {item['index'].get('error')}")
                
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                continue  # Continue with next batch even if this one failed

        def _prepare_colbert_vector(self, colbert_embedding, attention_mask):
            """
            Prepare ColBERT embedding as a single dense vector for Elasticsearch
            
            Args:
                colbert_embedding: ColBERT embedding from model
                attention_mask: Attention mask indicating which tokens are valid
                
            Returns:
                Single vector representation (e.g., mean of token embeddings)
            """
            # Get only the embeddings with attention mask = 1
            seq_len = colbert_embedding.shape[1]
            valid_embeddings = []
            
            for i in range(seq_len):
                if attention_mask[0, i] > 0:
                    valid_embeddings.append(colbert_embedding[0, i])
            
            if not valid_embeddings:
                # Fallback if no valid tokens
                return np.zeros(colbert_embedding.shape[2]).tolist()
            
            # Average the valid token embeddings
            mean_embedding = np.mean(valid_embeddings, axis=0)
            
            # Convert to list of floats for Elasticsearch
            return mean_embedding.tolist()
        
    # If we want to go at the token level..
    def _prepare_colbert_nested_vectors(self, colbert_embedding, attention_mask):
        """
        Prepare ColBERT token embeddings as nested objects for Elasticsearch
        As Colbert can have an encoding per token.
        
        Args:
            colbert_embedding: ColBERT embedding from model
            attention_mask: Attention mask indicating which tokens are valid
            
        Returns:
            List of dictionaries with token and vector fields
        """
        # ColBERT embeddings have shape [batch_size, seq_len, dim]
        # We need to convert to a list of {token, vector} objects
        seq_len = colbert_embedding.shape[1]
        dim = colbert_embedding.shape[2]
        
        nested_vectors = []
        
        # For each token position
        for i in range(seq_len):
            # Only include tokens that have a non-zero attention mask value
            if attention_mask[0, i] > 0:
                # Add token embedding as a nested object
                nested_vectors.append({
                    "token": f"token_{i}",  # We don't have actual tokens, just positions
                    "vector": colbert_embedding[0, i].tolist()
                })
        
        return nested_vectors

    def process_product_embeddings(self, products, clip_embedding, colbert_embedding=None):
        """
        Process product descriptions and images to create embeddings in batches
        
        Args:
            products: Collection of product dictionaries to process
            clip_embedding: CLIP embedding model instance
            colbert_embedding: Optional ColBERT embedding model instance
        """
        # Convert to list if it's not already
        products_list = list(products.values()) if isinstance(products, dict) else list(products)
        
        print(f"Processing embeddings for {len(products_list)} products in batches of {self.batch_size}")
        
        total_processed = 0
        total_images_processed = 0
        total_descriptions_processed = 0
        total_colbert_processed = 0
        
        for i in range(0, len(products_list), self.batch_size):
            batch = products_list[i:i + self.batch_size]
            processed_batch = []
            
            for product in batch:
                try:
                    # Generate description embedding if description exists
                    if 'description' in product and product.get('description'):
                        combined_text = f"{product['name']} {product['description']}".strip()
                        try:
                            # Get CLIP text embedding
                            clip_text_embedding = clip_embedding.get_text_embedding(combined_text)
                            product['description_clip_embedding'] = clip_text_embedding.embedding.tolist()[0]
                            total_descriptions_processed += 1
                            
                        except Exception as desc_err:
                            print(f"Error processing clip description emb for product {product['id']}: {desc_err}")

                        # Get ColBERT embeddings if model is provided
                        if colbert_embedding:
                            try:
                                colbert_text_embedding = colbert_embedding.get_colbert_sentence_embedding(combined_text)
                                product['description_embedding'] = colbert_text_embedding.embedding.tolist()[0]
                                total_colbert_processed += 1
                            except Exception as colbert_err:
                                self.logger.info(f"Error processing ColBERT embedding for product {product['id']}: {colbert_err}")
                    
                    # Generate image embedding if thumbnail exists and is accessible
                    if 'thumbnail_url' in product and product.get('thumbnail_url'):
                        try:
                            # Download image to temporary file
                            temp_image_path = process_image_url(product['thumbnail_url'])
                            if temp_image_path:
                                try:
                                    clip_image_embedding = clip_embedding.get_image_embedding(temp_image_path)
                                    product['image_embedding'] = clip_image_embedding.embedding.tolist()[0]
                                    total_images_processed += 1
                                finally:
                                    # Clean up temporary file
                                    if os.path.exists(temp_image_path):
                                        os.unlink(temp_image_path)
                        except Exception as img_err:
                            self.logger.info(f"Error processing image for product {product['id']}: {img_err}")
                    
                    processed_batch.append(product)
                    
                except Exception as e:
                    self.logger.info(f"Error processing embeddings for product {product['id']}: {e}")
            
            # Bulk index the processed batch
            if processed_batch:
                try:
                    # Prepare the bulk operations
                    bulk_operations = []
                    for product in processed_batch:
                        # Add the operation metadata
                        bulk_operations.append({
                            "index": {
                                "_index": self.index_name,
                                "_id": product['id']
                            }
                        })
                        # Add the document to index
                        bulk_operations.append(product)
                    
                    # Execute bulk operation with body parameter
                    results = self.es.bulk(body=bulk_operations, refresh=True)
                    
                    success = sum(1 for item in results['items'] if item['index']['status'] in [200, 201])
                    failed = len(results['items']) - success
                    
                    total_processed += success
                    progress = (i + len(batch)) / len(products_list) * 100
                    
                    self.logger.info(f"Batch {i//self.batch_size + 1}: Processed and indexed {success} products, "
                        f"{failed} failed ({progress:.1f}% complete)")
                    
                    # Print any indexing errors
                    if failed > 0:
                        for item in results['items']:
                            if item['index']['status'] not in [200, 201]:
                                print(f"Error indexing document {item['index'].get('_id')}: "
                                    f"{item['index'].get('error')}")
                    
                except Exception as e:
                    self.logger.info(f"Error bulk indexing batch with embeddings starting at index {i}: {e}")
                    self.logger.info("First few operations for debugging:", bulk_operations[:4] if bulk_operations else "No operations")

        self.logger.info(f"\nProcessing complete:")
        self.logger.info(f"Total products processed: {total_processed}")
        self.logger.info(f"Total images processed: {total_images_processed}")
        self.logger.info(f"Total descriptions processed: {total_descriptions_processed}")
        self.logger.info(f"Total ColBERT embeddings processed: {total_colbert_processed}")