#!/usr/bin/env python3
import os
import sys
import requests
import json
import time
from pprint import pprint

# Add parent directory to path to make imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import our modules - this will fail if run outside the project
try:
    from core.colbert_embeddings import ColBERTEmbedding
    from core.clip_embeddings import CLIPEmbedding
    local_imports = True
except ImportError:
    local_imports = False
    print("Running in API testing mode (no local imports available)")

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

def test_colbert_embedding():
    """Test the ColBERT embedding directly"""
    if not local_imports:
        print("Skipping local embedding test - running in API-only mode")
        return
    
    print("\n=== Testing ColBERT Embedding ===")
    
    try:
        # Initialize the embedding model
        print("Initializing ColBERT model...")
        colbert = ColBERTEmbedding()
        
        # Test queries
        test_queries = [
            "digital art tutorials",
            "sample packs for music production",
            "poker strategy guide",
            "learn javascript programming"
        ]
        
        # Test documents
        test_documents = [
            "Digital art tutorials for beginners. Learn how to create amazing artwork with our step-by-step guides.",
            "Professional sample packs with high-quality sounds for music producers. Includes drums, synths, and more.",
            "Advanced poker strategy guide with tips and tricks from professional players. Improve your game today!",
            "Complete JavaScript programming course for beginners. Learn to build web applications from scratch."
        ]
        
        print("\nGenerating embeddings...")
        query_embeddings = []
        document_embeddings = []
        
        for query in test_queries:
            embedding = colbert.get_colbert_embedding(query)
            query_embeddings.append(embedding)
            print(f"Query: '{query}' → Shape: {embedding.embedding.shape}, Tokens: {embedding.token_count}")
        
        for document in test_documents:
            embedding = colbert.get_colbert_embedding(document)
            document_embeddings.append(embedding)
            print(f"Document (shortened): '{document[:40]}...' → Shape: {embedding.embedding.shape}, Tokens: {embedding.token_count}")
        
        print("\nCalculating similarity scores...")
        
        # Calculate similarities for all query-document pairs
        similarity_matrix = []
        for i, q_emb in enumerate(query_embeddings):
            similarities = []
            for j, d_emb in enumerate(document_embeddings):
                sim = colbert.compute_similarity(
                    q_emb.embedding, 
                    d_emb.embedding,
                    query_mask=q_emb.attention_mask,
                    doc_mask=d_emb.attention_mask
                )
                similarities.append(sim)
            similarity_matrix.append(similarities)
        
        # Print similarity matrix
        print("\nSimilarity Matrix (normalized):")
        print("Rows = Queries, Columns = Documents")
        print("=" * 60)
        
        # Header
        print(f"{'Query \\ Document':<20}", end="")
        for j in range(len(test_documents)):
            print(f"Doc {j+1:<3}", end="  ")
        print()
        
        # Values
        for i, row in enumerate(similarity_matrix):
            print(f"Query {i+1}: {test_queries[i][:15]:<15}", end="")
            for sim in row:
                print(f"{sim:.3f}", end="  ")
            print()
        
        print("\nTest completed successfully!")
    
    except Exception as e:
        print(f"Error testing ColBERT embedding: {e}")
        raise

def test_api_endpoints():
    """Test the API endpoints for search"""
    print("\n=== Testing API Endpoints ===")
    
    # Check if API is accessible
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API is accessible")
            print(f"Health status: {health_response.json()}")
        else:
            print(f"⚠️ API returned error: {health_response.status_code}")
            return
    except Exception as e:
        print(f"⚠️ Could not connect to API: {e}")
        return
    
    # Test queries
    test_queries = [
        "digital art tutorials",
        "sample packs for music production",
        "poker strategy guide",
        "learn javascript programming"
    ]
    
    # Test different search endpoints
    endpoints = {
        "text_based": "/search_text_based",  # CLIP text
        "vision": "/search_vision",  # CLIP image
        "weighted_colbert": "/search_weighted_colbert"  # Our new weighted endpoint
    }
    
    results = {}
    
    print("\nTesting search endpoints with different queries...")
    
    for endpoint_name, endpoint_path in endpoints.items():
        print(f"\nTesting endpoint: {endpoint_name}")
        endpoint_results = {}
        
        for query in test_queries:
            try:
                print(f"  Query: '{query}'")
                
                # Prepare request
                url = f"{API_BASE_URL}{endpoint_path}"
                payload = {
                    "query": query,
                    "k": 5,
                    "num_candidates": 50
                }
                headers = {"Content-Type": "application/json"}
                
                # Make request and time it
                start_time = time.time()
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                query_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    results_count = len(data)
                    print(query_time, results_count, data)
                else:
                    print("no response")
            except Exception as e:
                print("err", e)