from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from elasticsearch import AsyncElasticsearch as Elasticsearch
from pydantic import BaseModel
from numba import jit

import json
import logging
import os
from contextlib import asynccontextmanager
import numpy as np
import torch
import time
import pathlib
import sys
import math

import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import re

# Add parent directory to path to make absolute imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports
from core.data_models import SearchQuery, SearchResponse, SearchResult, HealthStatus
from core.clip_embeddings import CLIPEmbedding
from core.colbert_embeddings import ColBERTEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


'''
    HI! I don't expect this to run in production, the code here meant to be quick and dirty. Hence the words "prototype".
    Given the write amount of time i'll clean up, setup up tests, modularize and comportemtalize the hell out it.
'''

# Global state
class AppState:
    def __init__(self):
        self.es_client = None
        self.clip_embed = None
        self.colbert_embed = None

app_state = AppState()

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        # Initialize Elasticsearch with standard client
        app_state.es_client = Elasticsearch(
            os.getenv("ELASTICSEARCH_URL", "http://es01:9200"),
            basic_auth=(
                os.getenv("ELASTICSEARCH_USERNAME", "elastic"),
                os.getenv("ELASTICSEARCH_PASSWORD", "changeme")
            )
        )
        # Initialize CLIP embeddings
        app_state.clip_embed = CLIPEmbedding()
        app_state.colbert_embed = ColBERTEmbedding()
        
        # Verify Elasticsearch connection
        if not await app_state.es_client.ping():
            logger.error("Failed to connect to Elasticsearch")
            raise Exception("Elasticsearch connection failed")
            
        # Log connection and license info
        try:
            license_info = await app_state.es_client.license.get()
            logger.info(f"Connected to Elasticsearch with license type: {license_info['license']['type']}")
            logger.info(f"License expiry date: {license_info['license']['expiry_date']}")
            
            # Check for Enterprise features
            if license_info['license']['type'] in ['trial', 'enterprise']:
                logger.info("Enterprise features are available")
            else:
                logger.warning("License does not support Enterprise features like RRF")
        except Exception as e:
            logger.warning(f"Failed to retrieve license info: {e}")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise e

    yield

    # Shutdown
    if app_state.es_client:
        await app_state.es_client.close()
        
    if app_state.clip_embed:
        del app_state.clip_embed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize FastAPI
app = FastAPI(
    title="CLIP Search and Embedding API",
    description="Combined semantic search and embedding service using CLIP",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_event():
    if app_state.es_client:
        await app_state.es_client.close()


# Dependencies
async def get_es_client():
    if app_state.es_client is None:  # Check without awaiting
        raise HTTPException(status_code=503, detail="Elasticsearch client not initialized")
    return app_state.es_client  # No need to await


async def get_clip():
    if not app_state.clip_embed:
        raise HTTPException(status_code=503, detail="CLIP model not initialized")
    return app_state.clip_embed

async def get_colbert():
    if not app_state.colbert_embed:
        raise HTTPException(status_code=503, detail="Colbert model not initialized")
    return app_state.colbert_embed

@app.post("/search_fuzzy", response_model=SearchResponse)
async def search_fuzzy(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip)):
    start_time = time.time()
    
    try:
        # Get text embedding for semantic search
        embedding_result = clip.get_text_embedding(query.query)
        query_embedding = embedding_result.embedding
        
        
        # Fuzzy text search
        text_response = await es_client.search(
            index='products',
            body={
                "_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"],
                "query": {
                    "bool": {
                        "should": [
                            # Fuzzy name matching with high boost
                            {
                                "fuzzy": {
                                    "name": {
                                        "value": query.query,
                                        "fuzziness": "AUTO",
                                        "max_expansions": 50,
                                        "prefix_length": 0,
                                        "boost": 3
                                    }
                                }
                            },
                            # Fuzzy description matching
                            {
                                "fuzzy": {
                                    "description": {
                                        "value": query.query,
                                        "fuzziness": 2,
                                        "max_expansions": 50,
                                        "prefix_length": 0,
                                        "boost": 1.5
                                    }
                                }
                            },
                            # Semantic vector search as fallback
                            {
                                "knn": {
                                    "field": "description_clip_embedding",
                                    "query_vector": query_embedding.tolist()[0],
                                    "k": query.k,
                                    "num_candidates": query.num_candidates,
                                    "boost": 2
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": query.k * 2,  # Get more candidates
                "collapse": {
                    "field": "native_type"  # Prevent too many similar results
                }
            }
        )
        
        # Process results
        results = []
        if text_response and 'hits' in text_response and 'hits' in text_response['hits']:
            for hit in text_response['hits']['hits']:
                print(hit)
                results.append(SearchResult(
                    score=hit['_score'],
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                    ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", ""),
                    seller_id=hit['_source'].get("seller", {}).get("id", ""),
                    seller_name=hit['_source'].get("seller", {}).get("name", ""),
                    seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
                ))
        
        # Sort and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:query.k]
        
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return SearchResponse(results=results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Fuzzy search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_text_based", response_model=SearchResponse)
async def search_text_based(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip)
):
    embedding_result = clip.get_text_embedding(query.query)
    query_embedding = embedding_result.embedding
    start_time = time.time()
    
    try:
        response = await es_client.search(
            index='products',
            body={
                "_source": [
                    "seller","description", "name", "thumbnail_url", "id","ratings", "price_cents", "url"],
                "knn": {
                    "field": "description_clip_embedding",
                    "query_vector": query_embedding.tolist()[0],
                    "k": query.k,
                    "num_candidates": query.num_candidates
                }
            },
            filter_path=['hits.hits._score', 'hits.hits._source']
        )

        # Process results
        results = []
        if response and 'hits' in response and 'hits' in response['hits']:
            for hit in response['hits']['hits']:
                results.append(SearchResult(
                    score=hit['_score'],
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
                ))
        
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return SearchResponse(results=results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Search endpoints
@app.post("/search_colbert", response_model=SearchResponse)
async def search_colbert(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    colbert: ColBERTEmbedding = Depends(get_colbert)
):
    """
    Perform semantic search using CLIP embeddings
    """
    start_time = time.time()
    
    try:
        # Get text embedding
        embedding_result = colbert.get_colbert_sentence_embedding(query.query)
        query_embedding = embedding_result.embedding
        
        print(query_embedding.tolist()[0])
        # Perform search
        response = await es_client.search(
            index='products',
            body={
                "_source": [
                    "seller","description", "name", "thumbnail_url", "id","ratings", "price_cents", "url"],
                "knn": {
                    "field": "description_embedding",
                    "query_vector": query_embedding.tolist()[0],
                    "k": query.k,
                    "num_candidates": query.num_candidates
                }
            },
            filter_path=['hits.hits._score', 'hits.hits._source']
        )
        
        # Process results
        results = []
        if response and 'hits' in response and 'hits' in response['hits']:
            for hit in response['hits']['hits']:
                results.append(SearchResult(
                    score=hit['_score'],
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
                ))
        
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return SearchResponse(results=results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar_vision", response_model=SearchResponse)
async def similar_vision(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip)
):
    """
    Perform semantic search using CLIP embeddings, with option to exclude a specific ID.
    Now includes ratings boost for more relevant results.
    """
    start_time = time.time()
    try:
        image_embedding = None
        query_embedding = None
        
        # Debug info
        logger.info(f"Received query: {query.dict()}")
        logger.info(f"ID present: {query.id is not None}")
        
        # Get embedding from ID if provided
        if query.id:
            logger.info(f"Fetching embeddings for ID: {query.id}")
            response = await es_client.search(
                index='products',
                body={
                    "_source": [
                    "seller","image_embedding", "description_clip_embedding"],
                    "query": {
                        "term": {
                            "_id": query.id
                        }
                    }
                }
            )
            
            # Debug the response
            logger.info(f"ID lookup response hit count: {len(response['hits']['hits']) if 'hits' in response and 'hits' in response['hits'] else 0}")
            
            if response and 'hits' in response and 'hits' in response['hits'] and len(response['hits']['hits']) > 0:
                source = response['hits']['hits'][0].get('_source', {})
                
                # Debug the source
                logger.info(f"Source fields: {list(source.keys())}")
                
                # Check if embeddings exist in the source
                image_embedding = source.get("image_embedding")
                query_embedding = source.get("description_clip_embedding")
                
                # Debug embedding extraction
                logger.info(f"Retrieved image_embedding: {'Yes' if image_embedding else 'No'}")
                logger.info(f"Retrieved description_clip_embedding: {'Yes' if query_embedding else 'No'}")
        
        # Fall back to generating embedding from the query text
        if not (query_embedding or image_embedding):
            logger.info(f"No embeddings found for ID, generating from query: {query.query}")
            embedding_result = clip.get_text_embedding(query.query)
            query_embedding = embedding_result.embedding.tolist()[0]
        
        # Use the best available embedding
        search_vector = image_embedding if image_embedding else query_embedding
        
        logger.info(f"Using search vector of type: {'image_embedding' if image_embedding else 'query_embedding'}")
        logger.info(f"Vector length: {len(search_vector) if search_vector else 0}")
        
        # Prepare the search body with bool query to exclude the specified ID
        search_body = {
            "_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"],
            "knn": {
                "field": "image_embedding",
                "query_vector": search_vector,
                "k": query.k,
                "num_candidates": query.num_candidates
            }
        }
        
        # Add bool query to exclude the specified ID if provided
        if query.id:
            search_body["query"] = {
                "bool": {
                    "must_not": [
                        {"term": {"_id": query.id}}
                    ]
                }
            }
        
        
        # Perform search
        response = await es_client.search(
            index='products',
            body=search_body
        )
        
        # Process results
        results = []
        if response and 'hits' in response and 'hits' in response['hits']:
            for hit in response['hits']['hits']:
                # Get ratings data
                ratings_count = hit['_source'].get("ratings", {}).get("count", 0)
                ratings_score = hit['_source'].get("ratings", {}).get("average", -1.0)
                
                # Get base score from elasticsearch
                base_score = hit.get('_score', 0)
                
                # Apply ratings boost for vision results (using "clip" config)
                final_score = apply_ratings_boost(base_score, ratings_count, ratings_score, "clip")
                # final_score = base_score
                results.append(SearchResult(
                    score=final_score,
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=ratings_count,
                    ratings_score=ratings_score,
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", ""),
                    id=hit.get('_id', '')
                ))
        
        # Sort by the new boosted score
        results.sort(key=lambda x: x.score, reverse=True)
        
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return SearchResponse(results=results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)  # Include full traceback
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoints
@app.post("/search_vision", response_model=SearchResponse)
async def search_vision(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip)
):
    """
    Perform semantic search using CLIP embeddings
    """
    start_time = time.time()
    
    try:
        # Get text embedding
        embedding_result = clip.get_text_embedding(query.query)
        query_embedding = embedding_result.embedding
        
        # Perform search
        response = await es_client.search(
            index='products',
            body={
                "_source": [
                    "seller",
                    "seller","description", "name", "thumbnail_url", "id","ratings", "price_cents", "url"],
                "knn": {
                    "field": "image_embedding",
                    "query_vector": query_embedding.tolist()[0],
                    "k": query.k,
                    "num_candidates": query.num_candidates
                }
            },
            filter_path=['hits.hits._score', 'hits.hits._source']
        )
        
        # Process results
        results = []
        if response and 'hits' in response and 'hits' in response['hits']:
            for hit in response['hits']['hits']:
                results.append(SearchResult(
                    score=hit['_score'],
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                    ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", "")
                ))
        
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return SearchResponse(results=results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoints
@app.post("/search_combined", response_model=SearchResponse)
async def search_combined(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip)
):
    """
    Perform semantic search using CLIP embeddings and RRF
    (Requires Enterprise license)
    """
    start_time = time.time()
    
    try:
        embedding_result = clip.get_text_embedding(query.query)
        query_embedding = embedding_result.embedding
        
        # Try RRF query first (Enterprise feature)
        try:
            response = await es_client.search(
                index='products',
                body={
                    "_source": [
                    "seller","description", "name", "thumbnail_url", "id","ratings", "price_cents", "url"],
                    "retriever": {
                        "rrf": {
                            "retrievers": [
                                {
                                    "standard": {
                                        "query": {
                                            "combined_fields": {
                                                "query": query.query,
                                                "fields": ["name^1.5", "description"],
                                                "fuzziness": 2,  # Number of character edits allowed
                                                "fuzzy_transpositions": True,  # Allow character swaps
                                                "prefix_length": 0,
                                                "max_expansions": 50
                                            }
                                        }
                                    }
                                },
                                {
                                    "knn": {
                                        "field": "image_embedding",
                                        "query_vector": query_embedding.tolist()[0],
                                        "k": query.k,
                                        "num_candidates": query.num_candidates
                                    }
                                }
                            ],
                            "rank_window_size": 50,
                            "rank_constant": 20
                        }
                    }
                },
                filter_path=['hits.hits._score', 'hits.hits._source']
            )
        except Exception as e:
            # If RRF fails (likely due to license), fall back to regular combined search
            logger.warning(f"RRF search failed, falling back to alternative combined search: {e}")
            return await search_lame_combined(query, es_client, clip)
        
        # Process results
        results = []
        if response and 'hits' in response and 'hits' in response['hits']:
            for hit in response['hits']['hits']:
                results.append(SearchResult(
                    score=hit['_score'],
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                    ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", "")
                ))
        
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return SearchResponse(results=results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_lame_combined", response_model=SearchResponse)
async def search_lame_combined(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip),
    colbert: ColBERTEmbedding = Depends(get_colbert)
):
    """
    Perform combined text and vector search with improved diversity and relevance
    """
    start_time = time.time()
    
    try:
        
        # Perform text search with more advanced query
        text_response = await es_client.search(
            index='products',
            body={
                "_source": [
                    "seller","description", "name", "thumbnail_url", "id","ratings", "price_cents", "url"],
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query.query,
                                    "fields": ["name^3", "description"],
                                    "type": "best_fields",
                                    "tie_breaker": 0.3
                                }
                            },
                            {
                                "match_phrase": {
                                    "name": {
                                        "query": query.query,
                                        "boost": 2
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": query.k * 2,  # Get more candidates
                "collapse": {
                    "field": "native_type"  # Prevent too many similar results
                }
            },
            filter_path=['hits.hits._score', 'hits.hits._source', 'hits.hits._id']
        )
        

        async def local_knn_response(query_embedding, field="description_clip_embedding"):
            knn_response = await es_client.search(
                index='products',
                body={
                    "_source": [
                    "seller","description", "name", "thumbnail_url", "id","ratings", "price_cents", "url"],
                    "knn": {
                        "field": field,  # Use description embedding
                        "query_vector": query_embedding.tolist()[0],
                        "k": query.k,
                        "num_candidates": query.num_candidates,
                        "filter": {
                            "bool": {
                                "must_not": [
                                    {"terms": {"id": [hit['_id'] for hit in text_response.get('hits', {}).get('hits', [])]}}
                                ]
                            }
                        }
                    }
                },
                filter_path=['hits.hits._score', 'hits.hits._source', 'hits.hits._id']
            )
            return knn_response
        
        # Combine and deduplicate results
        seen_ids = set()
        combined_results = []
        
        # Helper function to process hits
        def process_hits(response_hits, weight=0.5, is_text_search=False):
            if not response_hits or 'hits' not in response_hits:
                return
                
            hits = response_hits.get('hits', [])
            for hit in hits:
                doc_id = hit.get('_id')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    # More sophisticated scoring
                    original_score = float(hit.get('_score', 0))
                    if is_text_search:
                        # Boost text search results based on word match
                        title_match_boost = 1.5 if query.query.lower() in hit.get('_source', {}).get('name', '').lower() else 1.0
                        score = original_score * weight * title_match_boost
                    else:
                        score = original_score * weight
                    
                    source = hit.get('_source', {})
                    combined_results.append(SearchResult(
                    score=score,
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                    ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", "")
                ))
        
        # Process both result sets with different weights and strategies
        process_hits(text_response.get('hits'), 0.7, is_text_search=True)

        clip_embedding = clip.get_text_embedding(query.query).embedding
        img_knn_response = await local_knn_response(clip_embedding, "image_embedding")

        colbert_embedding = colbert.get_colbert_sentence_embedding(query.query).embedding
        txt_knn_response = await local_knn_response(colbert_embedding, "description_embedding")

        process_hits(txt_knn_response.get('hits'), 0.5)
        process_hits(img_knn_response.get('hits'), 0.3)
        
        # Sort by score and take top k results
        combined_results.sort(key=lambda x: x.score, reverse=True)
        combined_results = combined_results[:query.k]
        
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return SearchResponse(results=combined_results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/exact_match", response_model=SearchResponse)
async def exact_match(
    query: SearchQuery, es_client: Elasticsearch = Depends(get_es_client)
):
    start_time = time.time()

    try:
        response = await es_client.search(
            index="products",
            body={
                "_source": [
                    "description",
                    "name",
                    "thumbnail_url",
                    "id",
                    "ratings",
                    "price_cents",
                    "url",
                ],
                "from": 0,
                "size": 10,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"name": {"query": query.query, "boost": 2.0}}},
                            {"match": {"description": query.query}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            },
            filter_path=["hits.hits._score", "hits.hits._source"],
        )

        hits = response.get("hits", {}).get("hits", [])

        results = [
            SearchResult(
                score=hit["_score"],
                name=hit["_source"].get("name", ""),
                description=hit["_source"].get("description", ""),
                thumbnail_url=hit["_source"].get("thumbnail_url", ""),
                ratings_count=hit["_source"].get("ratings", {}).get("count", 0),
                ratings_score=hit["_source"].get("ratings", {}).get("average", -1.0),
                price_cents=hit["_source"].get("price_cents", 0),
                url=hit["_source"].get("url", ""),
                seller_id=hit['_source'].get("seller", {}).get("id", ""),
                seller_name=hit['_source'].get("seller", {}).get("name", ""),
                seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
            )
            for hit in hits
        ]

        return SearchResponse(
            results=results,
            query_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def get_fuzziness(query_text):
    length = len(query_text)
    if length <= 3:
        return 0
    elif length <= 8:
        return 1
    else:
        return 2
    
# def normalize_score(score, max_score, min_val=0.3, max_val=1.0):
#     if max_score == 0:
#         return min_val
#     return min(max_val, max(min_val, score / max_score))
    
@app.post("/search_combined_optimized_simplified", response_model=SearchResponse)
async def search_combined_optimized(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip),
    colbert: ColBERTEmbedding = Depends(get_colbert)
):
    start_time = time.time()
    min_acceptable_score = 0.4
    high_score_threshold = 0.85  # If BM25/Fuzzy score exceeds this, skip ColBERT/CLIP
    results = []
    seen_ids = set()

    executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_event_loop()
    
    # Start CLIP computation in background
    clip_future = loop.run_in_executor(
        executor, lambda: clip.get_text_embedding(query.query).embedding.tolist()[0]
    )

    async def bm25_fuzzy_search():
        body = {
            "_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"],
            "query": {
                "bool": {
                    "should": [
                        {"combined_fields": {"query": query.query, "fields": ["name^3", "description"]}},
                        {"fuzzy": {"name": {"value": query.query, "fuzziness": 1, "boost": 2.0}}},
                        {"fuzzy": {"description": {"value": query.query, "fuzziness": 1}}},
                    ]
                }
            },
            "size": query.k,
            "track_scores": True
        }
        response = await es_client.search(index='products', body=body)
        hits = response['hits']['hits']
        return hits if hits else []

    async def colbert_search():
        colbert_vector = colbert.get_colbert_sentence_embedding(query.query).embedding.tolist()[0]
        body = {
            "_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"],
            "knn": {
                "field": "description_embedding",
                "query_vector": colbert_vector,
                "k": query.k,
                "num_candidates": query.num_candidates // 2  # Reduce candidates for speed
            }
        }
        response = await es_client.search(index='products', body=body)
        return response['hits']['hits'] if response['hits']['hits'] else []

    async def clip_search():
        try:
            clip_vector = await asyncio.wait_for(clip_future, timeout=0.5)
        except asyncio.TimeoutError:
            clip_vector = clip.get_text_embedding(query.query).embedding.tolist()[0]
        
        body = {
            "_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"],
            "knn": {
                "field": "image_embedding",
                "query_vector": clip_vector,
                "k": query.k,
                "num_candidates": query.num_candidates // 2  # Reduce candidates
            }
        }
        response = await es_client.search(index='products', body=body)
        return response['hits']['hits'] if response['hits']['hits'] else []
    
    # Phase 1: BM25 + Fuzzy Search
    bm25_fuzzy_results = await bm25_fuzzy_search()
    max_bm25_fuzzy_score = max((hit['_score'] for hit in bm25_fuzzy_results), default=0)
    
    for hit in bm25_fuzzy_results:
        doc_id = hit['_id']
        seen_ids.add(doc_id)
        score = hit['_score'] / (max_bm25_fuzzy_score * 1.2)
        if query.query.lower() in hit['_source'].get('name', '').lower():
            score = min(1.0, score * 1.5)
        results.append(SearchResult(
                    score=score,
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                    ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", ""),
                    seller_id=hit['_source'].get("seller", {}).get("id", ""),
                    seller_name=hit['_source'].get("seller", {}).get("name", ""),
                    seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
                ))
    
    # Early exit if BM25/Fuzzy results are strong
    if max_bm25_fuzzy_score >= high_score_threshold:
        return SearchResponse(results=sorted(results, key=lambda x: x.score, reverse=True)[:query.k], query_time_ms=(time.time() - start_time) * 1000)
    
    # Phase 2: ColBERT Search
    colbert_results = await colbert_search()
    for hit in colbert_results:
        doc_id = hit['_id']
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            score = min(0.7, max(0.3, hit['_score']))
            results.append(SearchResult(
                    score=score,
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                    ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", ""),
                    seller_id=hit['_source'].get("seller", {}).get("id", ""),
                    seller_name=hit['_source'].get("seller", {}).get("name", ""),
                    seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
                ))
    
    # Phase 3: CLIP Search (only if needed)
    if not results or results[0].score < min_acceptable_score:
        clip_results = await clip_search()
        for hit in clip_results:
            doc_id = hit['_id']
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                score = min(0.6, max(0.2, hit['_score']))
                results.append(SearchResult(
                    score=score,
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                    ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", ""),
                    seller_id=hit['_source'].get("seller", {}).get("id", ""),
                    seller_name=hit['_source'].get("seller", {}).get("name", ""),
                    seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
                ))
    
    results.sort(key=lambda x: x.score, reverse=True)
    return SearchResponse(results=results[:query.k], query_time_ms=(time.time() - start_time) * 1000)

@app.post("/search_combined_v0_7", response_model=SearchResponse)
async def search_combined_v0_7(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip),
    colbert: ColBERTEmbedding = Depends(get_colbert)
):
    start_time = time.time()
    min_acceptable_score = 0.4
    results = []
    seen_ids = set()

    # Async CLIP computation
    executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_event_loop()
    clip_future = loop.run_in_executor(
        executor, 
        lambda: clip.get_text_embedding(query.query).embedding.tolist()[0]
    )

    # BM25 + Fuzzy Matching
    
    bm25_query = {
        "_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"],
        "query": {
            "bool": {
                "should": [
                    {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "OR"}},
                    {"fuzzy": {"name": {"value": query.query, "fuzziness": get_fuzziness(query.query), "prefix_length": 1, "boost": 2.0}}}
                ]
            }
        },
        "size": query.k
    }
    bm25_response = await es_client.search(index='products', body=bm25_query)
    bm25_hits = bm25_response['hits']['hits']
    max_score_bm25 = max([hit.get('_score', 0) for hit in bm25_hits]) if bm25_hits else 1.0
    
    for hit in bm25_hits:
        doc_id = hit.get('_id')
        seen_ids.add(doc_id)
        results.append(SearchResult(
                score=normalize_score(hit.get('_score', 0), max_score_bm25),
                base_score=hit.get('_score', 0),
                score_origin="bm25",
                name=hit['_source'].get('name', ''),
                description=hit['_source'].get('description', ''),
                thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                price_cents=hit['_source'].get("price_cents", 0),
                url=hit['_source'].get("url", ""),
                id=doc_id,
                seller_id=hit['_source'].get("seller", {}).get("id", ""),
                seller_name=hit['_source'].get("seller", {}).get("name", ""),
                seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
            ))
        

    # ColBERT Semantic Search
    colbert_vector = colbert.get_colbert_sentence_embedding(query.query).embedding.tolist()[0]
    colbert_query = {"_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"], "knn": {"field": "description_embedding", "query_vector": colbert_vector, "k": query.k, "num_candidates": query.num_candidates}}
    colbert_response = await es_client.search(index='products', body=colbert_query)
    colbert_hits = colbert_response['hits']['hits']
    
    for hit in colbert_hits:
        doc_id = hit.get('_id')
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            results.append(SearchResult(
                score=normalize_score(hit.get('_score', 0), 1.0),
                base_score=hit.get('_score', 0),
                score_origin="colbert",
                name=hit['_source'].get('name', ''),
                description=hit['_source'].get('description', ''),
                thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                price_cents=hit['_source'].get("price_cents", 0),
                url=hit['_source'].get("url", ""),
                id=doc_id,
                seller_id=hit['_source'].get("seller", {}).get("id", ""),
                seller_name=hit['_source'].get("seller", {}).get("name", ""),
                seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
            ))
    
    # CLIP Vision Search (If Needed)
    if not results or results[0].score < min_acceptable_score:
        clip_vector = await asyncio.wait_for(clip_future, timeout=0.5)
        clip_query = {"_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"], "knn": {"field": "image_embedding", "query_vector": clip_vector, "k": query.k, "num_candidates": query.num_candidates}}
        clip_response = await es_client.search(index='products', body=clip_query)
        
        for hit in clip_response['hits']['hits']:
            doc_id = hit.get('_id')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append(SearchResult(
                    score=normalize_score(hit.get('_score', 0), 0.6),
                    base_score=hit.get('_score', 0),
                    score_origin="bm25",
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=hit['_source'].get("ratings", {}).get("count", 0),
                    ratings_score=hit['_source'].get("ratings", {}).get("average", -1.0),
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", ""),
                    id=doc_id,
                    seller_id=hit['_source'].get("seller", {}).get("id", ""),
                    seller_name=hit['_source'].get("seller", {}).get("name", ""),
                    seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
                ))
    
    results.sort(key=lambda x: x.score, reverse=True)
    query_time = (time.time() - start_time) * 1000
    return SearchResponse(results=results[:query.k], query_time_ms=query_time)



# @TODO Need to setup a regression task or optuna) to obtain optimal relevance_threshold, and boost..
# Centralized configuration for ratings boost parameters
@app.post("/search_combined_v0_8", response_model=SearchResponse)
async def search_combined_optimal(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip),
    colbert: ColBERTEmbedding = Depends(get_colbert)
):
    start_time = time.time()
    min_acceptable_score = 0.4  # Same threshold as v0.7
    results = []
    seen_ids = set()

    # Start CLIP embedding computation asynchronously right away
    # (will only be used if BM25 and ColBERT don't produce good results)
    executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_event_loop()
    clip_future = loop.run_in_executor(
        executor, 
        lambda: clip.get_text_embedding(query.query).embedding.tolist()[0]
    )
    
    # Get ColBERT embedding - this is fast and needed for the query
    colbert_vector = colbert.get_colbert_sentence_embedding(query.query).embedding.tolist()[0]
    
    # Prepare BM25 query
    bm25_query = {
        "_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"],
        "query": {
            "bool": {
                "should": [
                    {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "OR"}},
                    {"fuzzy": {"name": {"value": query.query, "fuzziness": get_fuzziness(query.query), "prefix_length": 1,}}}
                ]
            }
        },
        "size": query.k
    }
    
    # Execute BM25 search
    bm25_response = await es_client.search(index='products', body=bm25_query)
    
    # Process BM25 results
    bm25_hits = bm25_response['hits']['hits']
    max_score_bm25 = max([hit.get('_score', 0) for hit in bm25_hits]) if bm25_hits else 1.0
    
    for hit in bm25_hits:
        doc_id = hit.get('_id')
        seen_ids.add(doc_id)
        
        # Get ratings data
        ratings_count = hit['_source'].get("ratings", {}).get("count", 0)
        ratings_score = hit['_source'].get("ratings", {}).get("average", -1.0)
        
        # Calculate base score (same as v0.7)
        # base_score = normalize_score(hit.get('_score', 0), max_score_bm25)

        normalized_score = 1-2/(1+hit.get('_score', 0))
        
        # Apply ratings boost for BM25 results
        final_score = apply_ratings_boost(normalized_score, ratings_count, ratings_score, "bm25")

        # final_score = normalize_score(final_score, 1)
        
        results.append(SearchResult(
            score=final_score,
            base_score=hit.get('_score', 0),
            score_origin="bm25",
            name=hit['_source'].get('name', ''),
            description=hit['_source'].get('description', ''),
            thumbnail_url=hit['_source'].get('thumbnail_url', ''),
            ratings_count=ratings_count,
            ratings_score=ratings_score,
            price_cents=hit['_source'].get("price_cents", 0),
            url=hit['_source'].get("url", ""),
            id=doc_id,
            seller_id=hit['_source'].get("seller", {}).get("id", ""),
            seller_name=hit['_source'].get("seller", {}).get("name", ""),
            seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
        ))
    
    # Execute ColBERT search - only if needed (mimicking v0.7 behavior)
    # But prepare it early to avoid wasting time
    colbert_query = {
        "_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"], 
        "knn": {
            "field": "description_embedding", 
            "query_vector": colbert_vector, 
            "k": query.k, 
            "num_candidates": query.num_candidates
        }
    }
    
    # ColBERT Semantic Search
    colbert_response = await es_client.search(index='products', body=colbert_query)
    colbert_hits = colbert_response['hits']['hits']
    
    for hit in colbert_hits:
        doc_id = hit.get('_id')
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            
            # Get ratings data
            ratings_count = hit['_source'].get("ratings", {}).get("count", 0)
            ratings_score = hit['_source'].get("ratings", {}).get("average", -1.0)
            
            # Calculate base score (same as v0.7)
            base_score = normalize_score(hit.get('_score', 0), 1.0)
            
            # Apply ratings boost for ColBERT results
            final_score = apply_ratings_boost(base_score, ratings_count, ratings_score, "colbert")
            
            results.append(SearchResult(
                score=final_score,
                base_score=base_score,
                score_origin="colbert",
                name=hit['_source'].get('name', ''),
                description=hit['_source'].get('description', ''),
                thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                ratings_count=ratings_count,
                ratings_score=ratings_score,
                price_cents=hit['_source'].get("price_cents", 0),
                url=hit['_source'].get("url", ""),
                id=doc_id,
                seller_id=hit['_source'].get("seller", {}).get("id", ""),
                seller_name=hit['_source'].get("seller", {}).get("name", ""),
                seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
            ))
    
    # Check if we need CLIP results
    best_score_so_far = max([r.score for r in results]) if results else 0
    
    if not results or best_score_so_far < min_acceptable_score:
        try:
            # Await the CLIP embedding that was started earlier
            clip_vector = await asyncio.wait_for(clip_future, timeout=0.5)
            
            # Execute CLIP search
            clip_query = {
                "_source": [
                    "seller","description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"], 
                "knn": {
                    "field": "image_embedding", 
                    "query_vector": clip_vector, 
                    "k": query.k, 
                    "num_candidates": query.num_candidates
                }
            }
            
            clip_response = await es_client.search(index='products', body=clip_query)
            
            # Process CLIP results
            for hit in clip_response['hits']['hits']:
                doc_id = hit.get('_id')
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    
                    # Get ratings data
                    ratings_count = hit['_source'].get("ratings", {}).get("count", 0)
                    ratings_score = hit['_source'].get("ratings", {}).get("average", -1.0)
                    
                    # Calculate base score (same as v0.7)
                    base_score = normalize_score(hit.get('_score', 0), 0.6)
                    
                    # Apply ratings boost for CLIP results
                    final_score = apply_ratings_boost(base_score, ratings_count, ratings_score, "clip")
                    
                    results.append(SearchResult(
                        score=final_score,
                        base_score=base_score,
                        score_origin="clip",
                        name=hit['_source'].get('name', ''),
                        description=hit['_source'].get('description', ''),
                        thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                        ratings_count=ratings_count,
                        ratings_score=ratings_score,
                        price_cents=hit['_source'].get("price_cents", 0),
                        url=hit['_source'].get("url", ""),
                        id=doc_id,
                        seller_id=hit['_source'].get("seller", {}).get("id", ""),
                        seller_name=hit['_source'].get("seller", {}).get("name", ""),
                        seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
                    ))
        except asyncio.TimeoutError:
            # CLIP embedding took too long, proceed without it
            pass
    
    # Sort and limit final results
    results.sort(key=lambda x: x.score, reverse=True)
    query_time = (time.time() - start_time) * 1000
    return SearchResponse(results=results[:query.k], query_time_ms=query_time)

# Updated RATINGS_BOOST_CONFIG to work with Wilson Score
BOOSTS_CONFIG = {
    "default": {
        "ratings_relevance_threshold": 0.7,  
        "ratings_boost_multiplier": 0.03     
    },
    "bm25": {
        "ratings_relevance_threshold": 0.75,
        "ratings_boost_multiplier": 0.51
    },
    "colbert": {
        "ratings_relevance_threshold": 0.8,
        "ratings_boost_multiplier": 0.05
    },
    "clip": {
        "ratings_relevance_threshold": 0.7,
        "ratings_boost_multiplier": 0.01
    }
}

def wilson_lower_bound(pos, n, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score with robust error handling
    
    Args:
    pos (int): Number of positive ratings
    n (int): Total number of ratings
    confidence (float): Confidence level (default 0.95)
    
    Returns:
    float: Wilson Lower Bound score, or 0 if calculation is impossible
    """
    # Handle edge cases
    if n == 0 or pos < 0 or pos > n:
        return 0.0
    
    # Confidence level z-scores
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    z = z_scores.get(confidence, 1.96)
    
    # Prevent division by zero and handle small sample sizes
    if n < 1:
        return 0.0
    
    try:
        pos_rate = pos / n
        
        # Prevent negative values inside sqrt
        inner_sqrt = max(0, pos_rate * (1 - pos_rate) + z*z / (4 * n))
        
        # Wilson score interval calculation with additional safeguards
        numerator = pos_rate + z*z / (2 * n) - z * math.sqrt(inner_sqrt / n)
        denominator = 1 + z*z / n
        
        # Ensure result is between 0 and 1
        return max(0, min(1, numerator / denominator))
    
    except (ValueError, ZeroDivisionError):
        # Fallback to 0 if any calculation fails
        return 0.0

def calculate_wilson_ratings_boost(ratings_count, ratings_score, max_ratings=500):
    """
    Calculate ratings boost using Wilson Lower Bound method with robust handling
    
    Args:
    ratings_count (int): Total number of ratings
    ratings_score (float): Average rating score
    max_ratings (int): Maximum ratings for normalization
    
    Returns:
    float: Normalized ratings boost
    """
    # Validate inputs
    if ratings_count <= 0 or ratings_score < 0 or ratings_score > 5:
        return 0.0
    
    # Convert rating score to number of positive ratings
    # Round to handle potential floating-point imprecision
    pos_ratings = round(ratings_score * ratings_count)
    
    # Ensure pos_ratings doesn't exceed total ratings
    pos_ratings = min(pos_ratings, ratings_count)
    
    # Calculate Wilson Lower Bound
    wilson_score = wilson_lower_bound(pos_ratings, ratings_count)
    
    # Logarithmic scaling of ratings count
    log_factor = math.log(1 + ratings_count) / math.log(1 + max_ratings)
    
    # Combine Wilson score with log scaling
    return 0.7 * wilson_score + 0.3 * log_factor

def apply_ratings_boost(base_score, ratings_count, ratings_score, search_method="default", base_score_weight=False):
    """
    Apply a ratings boost to a base score based on configured parameters for each search method.
    
    Parameters:
    base_score (float): The base relevance score
    ratings_count (int): Number of ratings
    ratings_score (float): Average rating score
    search_method (str): Which search method is being used ("bm25", "colbert", "clip" or "default")
    
    Returns:
    float: The final score with ratings boost applied
    """
    # Get configuration for this search method, falling back to default if not specified
    config = BOOSTS_CONFIG.get(search_method, BOOSTS_CONFIG["default"])
    relevance_threshold = config["ratings_relevance_threshold"]
    boost_multiplier = config["ratings_boost_multiplier"]
    
    # Only apply boost if base score meets threshold and ratings exist
    if base_score < relevance_threshold or ratings_count <= 0 or ratings_score < 0:
        return base_score
    
    # Calculate ratings boost using Wilson Lower Bound method
    ratings_boost = calculate_wilson_ratings_boost(ratings_count, ratings_score)
    
    # Apply configured multiplier and add to base score
    return base_score + (boost_multiplier * ratings_boost)


def normalize_score(score, max_score):
    """Normalize score to a 0-1 range and apply sigmoid to bias towards higher scores"""
    normalized = score / max_score if max_score > 0 else 0
    return 1 / (1 + math.exp(-10 * (normalized - 0.5)))


class CachedEmbeddingService:
    def __init__(self, colbert_service):
        self.colbert_service = colbert_service
        self.cache = {}
        self.max_cache_size = 1000
        
    def get_embedding(self, query_text):
        if query_text in self.cache:
            return self.cache[query_text]
        
        # Calculate new embedding
        embedding = self.colbert_service.get_colbert_sentence_embedding(query_text).embedding.tolist()[0]
        
        # Add to cache, removing oldest item if needed
        if len(self.cache) >= self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[query_text] = embedding
        return embedding

# FastAPI dependency
def get_cached_embedding_service(colbert: ColBERTEmbedding = Depends(get_colbert)):
    return CachedEmbeddingService(colbert)

@jit(nopython=True)
def batch_cosine_similarity(query_vector, doc_vectors):
    """
    Vectorized dot product for pre-normalized vectors.
    """
    # One line, fully vectorized dot product
    return np.dot(doc_vectors, query_vector)

@app.post("/two_phase_unnative", response_model=SearchResponse)
async def two_phase_unnative(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    colbert: ColBERTEmbedding = Depends(get_colbert),
    embedding_service: CachedEmbeddingService = Depends(get_cached_embedding_service)
):
    start_time = time.time()
    
    # Reduce _source fields to only what's needed
    bm25_query = {
        "_source": ["name", "description", "thumbnail_url", "id", 
                   "ratings.count", "ratings.average", "price_cents", "url", 
                   "seller.id", "seller.name", "seller.avatar_url"],
        "query": {
            "bool": {
                "should": [
                    {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "OR"}},
                    {"match_phrase": {"combined_text": {"query": query.query, "boost": 1.0}}},
                    {"fuzzy": {"name": {"value": query.query, "fuzziness": get_fuzziness(query.query), "boost": 1.0}}}
                ]
            }
        },
        "size": min(100, query.num_candidates)
    }
    
    # Execute BM25 search
    bm25_response = await es_client.search(index='products', body=bm25_query)
    candidates = bm25_response['hits']['hits']
    
    if not candidates:
        return SearchResponse(results=[], query_time_ms=0)
    
    max_bm25_score = max([hit.get('_score', 0) for hit in candidates])
    
    # Get candidate IDs for batch embedding retrieval
    candidate_ids = [hit['_id'] for hit in candidates]
    
    # Get query embedding - with caching
    try:
        query_vector = embedding_service.get_query_embedding(query.query)
    except Exception as e:
        # Fallback to direct call if needed
        query_vector = colbert.get_colbert_sentence_embedding(query.query).embedding.tolist()[0]
    
    # Fetch document embeddings in bulk
    embeddings_query = {
        "_source": ["description_embedding"],
        "query": {
            "ids": {
                "values": candidate_ids
            }
        },
        "size": len(candidate_ids)
    }
    
    # Create mapping for document data while waiting for embeddings
    doc_data = {hit['_id']: hit for hit in candidates}
    
    # Execute search with await (using async ES client directly)
    embeddings_response = await es_client.search(index='products', body=embeddings_query)
    
    # Process document embeddings
    doc_embeddings = {}
    embedding_arrays = []
    valid_doc_ids = []
    
    # First pass: extract document embeddings
    for hit in embeddings_response['hits']['hits']:
        doc_id = hit['_id']
        embedding = hit['_source'].get('description_embedding')
        if embedding:
            doc_embeddings[doc_id] = embedding
            embedding_arrays.append(embedding)
            valid_doc_ids.append(doc_id)
    
    # Batch process similarity calculation using Numba
    similarity_scores = {}
    
    if embedding_arrays:
        # Convert to numpy arrays
        query_vector_array = np.array(query_vector, dtype=np.float32)
        doc_vectors_array = np.array(embedding_arrays, dtype=np.float32)
        
        # Calculate all similarities at once with optimized Numba function
        all_similarities = batch_cosine_similarity(query_vector_array, doc_vectors_array)
        
        # Map back to document IDs
        for doc_id, sim_score in zip(valid_doc_ids, all_similarities):
            similarity_scores[doc_id] = float(sim_score)
    
    # Determine if this is a multi-term query
    is_multi_term = len([t for t in query.query.split() if len(t) > 3]) > 1
    bm25_weight = 0.4 if is_multi_term else 0.6
    colbert_weight = 0.6 if is_multi_term else 0.4
    
    # Process results
    reranked_results = []
    for doc_id in candidate_ids:
        hit = doc_data[doc_id]
        
        # Get ratings data
        ratings = hit['_source'].get('ratings', {})
        ratings_count = ratings.get('count', 0) if isinstance(ratings, dict) else 0
        ratings_score = ratings.get('average', -1.0) if isinstance(ratings, dict) else -1.0
        
        # Normalize BM25 score
        normalized_bm25 = 1 - 2 / (1 + hit.get('_score', 0))
        
        # Get colbert similarity score
        colbert_score = similarity_scores.get(doc_id, 0)
        
        # Calculate final score
        final_score = (normalized_bm25 * bm25_weight) + (colbert_score * colbert_weight)
        
        # Apply ratings boost
        final_score = apply_ratings_boost(final_score, ratings_count, ratings_score, "hybrid")
        
        # Get seller data
        seller = hit['_source'].get('seller', {})
        seller_id = seller.get('id', '') if isinstance(seller, dict) else ''
        seller_name = seller.get('name', '') if isinstance(seller, dict) else ''
        seller_avatar = seller.get('avatar_url', '') if isinstance(seller, dict) else ''
        
        # Create result
        reranked_results.append(SearchResult(
            score=final_score,
            base_score=hit.get('_score', 0),
            score_origin="hybrid",
            name=hit['_source'].get('name', ''),
            description=hit['_source'].get('description', ''),
            thumbnail_url=hit['_source'].get('thumbnail_url', ''),
            ratings_count=ratings_count,
            ratings_score=ratings_score,
            price_cents=hit['_source'].get('price_cents', 0),
            url=hit['_source'].get('url', ''),
            id=doc_id,
            seller_id=seller_id,
            seller_name=seller_name,
            seller_thumbnail_url=seller_avatar
        ))
    
    # Sort and limit results
    reranked_results.sort(key=lambda x: x.score, reverse=True)
    query_time = (time.time() - start_time) * 1000
    
    return SearchResponse(results=reranked_results[:query.k], query_time_ms=query_time)



from fastapi.responses import StreamingResponse


@app.post("/two_phase_stream")
async def two_phase_stream(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    colbert: ColBERTEmbedding = Depends(get_colbert)
):
    async def generate():
        start_time = time.time()
        
        # First phase: BM25 search (deliver immediately)
        bm25_query = {
            "_source": ["name", "description", "thumbnail_url", "id", "ratings", "price_cents", "url", "seller"],
            "query": {
                "bool": {
                    "should": [
                        {"combined_fields": {"query": query.query, "fields": ["name^3", "description"]}},
                        {"match_phrase": {"combined_text": {"query": query.query, "boost": 1.0}}},
                        {"fuzzy": {"name": {"value": query.query, "fuzziness": get_fuzziness(query.query), "boost": 1.0}}}
                    ]
                }
            },
            "size": min(20, query.k)
        }
        
        bm25_response = await es_client.search(index="products", body=bm25_query)
        
        # Process and yield first batch
        first_batch = []
        candidate_ids = []
        doc_data = {}
        
        # Store the max score for normalizing
        max_bm25_score = max([hit.get('_score', 0) for hit in bm25_response["hits"]["hits"]]) if bm25_response["hits"]["hits"] else 1.0
        
        for hit in bm25_response["hits"]["hits"]:
            # Store the full hit data for reranking later
            doc_data[hit["_id"]] = hit
            candidate_ids.append(hit["_id"])
            
            # Get ratings data
            ratings = hit["_source"].get("ratings", {})
            ratings_count = ratings.get("count", 0) if isinstance(ratings, dict) else 0
            ratings_score = ratings.get("average", -1.0) if isinstance(ratings, dict) else -1.0
            
            # Get seller data
            seller = hit["_source"].get("seller", {})
            seller_id = seller.get("id", "") if isinstance(seller, dict) else ""
            seller_name = seller.get("name", "") if isinstance(seller, dict) else ""
            seller_avatar = seller.get("avatar_url", "") if isinstance(seller, dict) else ""
            
            # Create search result
            result = SearchResult(
                score=hit.get("_score", 0) / max_bm25_score,
                base_score=hit.get("_score", 0),
                score_origin="bm25",
                name=hit["_source"].get("name", ""),
                description=hit["_source"].get("description", ""),
                thumbnail_url=hit["_source"].get("thumbnail_url", ""),
                ratings_count=ratings_count,
                ratings_score=ratings_score,
                price_cents=hit["_source"].get("price_cents", 0),
                url=hit["_source"].get("url", ""),
                id=hit["_id"],
                seller_id=seller_id,
                seller_name=seller_name,
                seller_thumbnail=seller_avatar
            )
            
            first_batch.append(result.dict())
        
        # Send first batch
        first_phase_time = (time.time() - start_time) * 1000
        yield json.dumps({
            "results": first_batch,
            "phase": "bm25",
            "query_time_ms": first_phase_time,
            "complete": False
        }) + "\n"
        
        # Second phase: Get embeddings and rerank
        try:
            # Only proceed with second phase if we have candidates
            if not candidate_ids:
                yield json.dumps({
                    "results": [],
                    "phase": "reranked",
                    "query_time_ms": first_phase_time,
                    "complete": True
                }) + "\n"
                return
            
            # Get query embedding
            query_embedding = colbert.get_colbert_sentence_embedding(query.query).embedding.tolist()[0]
            
            # Fetch all embeddings in batch
            embeddings_response = await es_client.search(
                index="products",
                body={
                    "_source": ["description_embedding"],
                    "query": {
                        "ids": {
                            "values": candidate_ids
                        }
                    },
                    "size": len(candidate_ids)
                }
            )
            
            # Process document embeddings
            embedding_arrays = []
            valid_doc_ids = []
            
            # Extract document embeddings
            for hit in embeddings_response["hits"]["hits"]:
                doc_id = hit["_id"]
                embedding = hit["_source"].get("description_embedding")
                if embedding:
                    embedding_arrays.append(embedding)
                    valid_doc_ids.append(doc_id)
            
            # Calculate similarity scores
            similarity_scores = {}
            
            if embedding_arrays:
                # Convert to numpy arrays
                query_vector_array = np.array(query_embedding, dtype=np.float32)
                doc_vectors_array = np.array(embedding_arrays, dtype=np.float32)
                
                # Calculate dot product (no need for Numba)
                all_similarities = np.dot(doc_vectors_array, query_vector_array)
                
                # Map back to document IDs
                for doc_id, sim_score in zip(valid_doc_ids, all_similarities):
                    similarity_scores[doc_id] = float(sim_score)
            
            # Determine if this is a multi-term query
            is_multi_term = len([t for t in query.query.split() if len(t) > 3]) > 1
            bm25_weight = 0.4 if is_multi_term else 0.6
            colbert_weight = 0.6 if is_multi_term else 0.4
            
            # Process results
            reranked_results = []
            
            for doc_id in candidate_ids:
                hit = doc_data[doc_id]
                
                # Get ratings data
                ratings = hit["_source"].get("ratings", {})
                ratings_count = ratings.get("count", 0) if isinstance(ratings, dict) else 0
                ratings_score = ratings.get("average", -1.0) if isinstance(ratings, dict) else -1.0
                
                # Normalize BM25 score
                normalized_bm25 = 1 - 2 / (1 + hit.get("_score", 0))
                
                # Get colbert similarity score
                colbert_score = similarity_scores.get(doc_id, 0)
                
                # Calculate final score
                final_score = (normalized_bm25 * bm25_weight) + (colbert_score * colbert_weight)
                
                # Apply ratings boost
                final_score = apply_ratings_boost(final_score, ratings_count, ratings_score, "hybrid")
                
                # Get seller data
                seller = hit["_source"].get("seller", {})
                seller_id = seller.get("id", "") if isinstance(seller, dict) else ""
                seller_name = seller.get("name", "") if isinstance(seller, dict) else ""
                seller_avatar = seller.get("avatar_url", "") if isinstance(seller, dict) else ""
                
                # Create result
                result = SearchResult(
                    score=final_score,
                    base_score=hit.get("_score", 0),
                    score_origin="hybrid",
                    name=hit["_source"].get("name", ""),
                    description=hit["_source"].get("description", ""),
                    thumbnail_url=hit["_source"].get("thumbnail_url", ""),
                    ratings_count=ratings_count,
                    ratings_score=ratings_score,
                    price_cents=hit["_source"].get("price_cents", 0),
                    url=hit["_source"].get("url", ""),
                    id=doc_id,
                    seller_id=seller_id,
                    seller_name=seller_name,
                    seller_thumbnail=seller_avatar
                )
                
                reranked_results.append(result.dict())
            
            # Sort results by score
            reranked_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit to requested k
            reranked_results = reranked_results[:query.k]
            
            # Send reranked results
            yield json.dumps({
                "results": reranked_results,
                "phase": "reranked",
                "query_time_ms": (time.time() - start_time) * 1000,
                "complete": True
            }) + "\n"
            
        except Exception as e:
            # Log the exception
            logger.error(f"Error in second phase: {str(e)}", exc_info=True)
            
            # Send error but client still has the first batch
            yield json.dumps({
                "error": str(e),
                "phase": "error",
                "complete": True
            }) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )




@app.post("/search_fallback", response_model=SearchResponse)
async def search_fallback(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client)
):
    """
    Fallback search that doesn't rely on vector embeddings at all - 
    pure text search with BM25 and fuzzy matching
    """
    start_time = time.time()
    
    try:
        # Perform text-only search
        response = await es_client.search(
            index='products',
            body={
                "_source": [
                    "seller","description", "name", "thumbnail_url", "id","ratings", "price_cents", "url"],
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query.query,
                                    "fields": ["name^3", "description"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO",
                                    "prefix_length": 2,
                                    "max_expansions": 50,
                                    "fuzzy_transpositions": True,
                                    "tie_breaker": 0.3
                                }
                            },
                            {
                                "match_phrase": {
                                    "name": {
                                        "query": query.query,
                                        "boost": 2.0
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": query.k
            }
        )
        
        # Process results
        results = []
        for hit in response['hits']['hits']:
            results.append(SearchResult(
                score=hit['_score'],
                name=hit['_source'].get('name', ''),
                description=hit['_source'].get('description', ''),
                thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                seller_id=hit['_source'].get("seller", {}).get("id", ""),
                seller_name=hit['_source'].get("seller", {}).get("name", ""),
                seller_thumbnail_url=hit['_source'].get("seller", {}).get("avatar_url", "")
            ))
        
        query_time = (time.time() - start_time) * 1000
        return SearchResponse(results=results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Fallback search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Health check endpoint
@app.get("/health", response_model=HealthStatus)
async def health_check(
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip)
):
    """Check the health status of the service"""
    es_health = es_client.ping()
    
    # Check license information
    license_info = "unknown"
    enterprise_features = False
    try:
        license_data = es_client.license.get()
        license_info = license_data['license']['type']
        enterprise_features = license_info in ['trial', 'enterprise']
    except Exception:
        pass
    
    return HealthStatus(
        status="healthy",
        elasticsearch=es_health,
        is_model_loaded=clip is not None,
        device=str(clip.device)
    )

# Endpoint to extend trial license
@app.post("/extend_trial")
async def extend_trial(
    es_client: Elasticsearch = Depends(get_es_client)
):
    """
    Attempt to extend the trial license
    """
    try:
        response = await es_client.license.post_start_trial(acknowledge=True)
        return {"success": response.get('acknowledged', False), "message": response.get('trial_was_started', False)}
    except Exception as e:
        logger.error(f"Error extending trial: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get license information
@app.get("/license")
async def get_license(
    es_client: Elasticsearch = Depends(get_es_client)
):
    """
    Get current license information
    """
    try:
        license_data = es_client.license.get()
        return {
            "type": license_data['license']['type'],
            "status": license_data['license']['status'],
            "expiry_date": license_data['license']['expiry_date'],
            "enterprise_features_available": license_data['license']['type'] in ['trial', 'enterprise']
        }
    except Exception as e:
        logger.error(f"Error getting license info: {e}")
        raise HTTPException(status_code=500, detail=str(e))