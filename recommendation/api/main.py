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
import requests

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
    Given the right amount of time i'll clean up, setup up tests, modularize and compartementalize the hell out it.
'''

# Global state
class AppState:
    def __init__(self):
        self.es_client = None
        self.clip_embed = None
        self.colbert_embed = None

app_state = AppState()


stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stopwords_set = set(stopwords_list.decode().splitlines()) 

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
                    seller_thumbnail=hit['_source'].get("seller", {}).get("avatar_url", "")
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
                seller_thumbnail=hit['_source'].get("seller", {}).get("avatar_url", "")
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
                    {"match_phrase": {"combined_text": {"query": query.query}}},
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
                seller_thumbnail=hit['_source'].get("seller", {}).get("avatar_url", "")
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
                seller_thumbnail=hit['_source'].get("seller", {}).get("avatar_url", "")
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
                    seller_thumbnail=hit['_source'].get("seller", {}).get("avatar_url", "")
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
            seller_thumbnail=hit['_source'].get("seller", {}).get("avatar_url", "")
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
                seller_thumbnail=hit['_source'].get("seller", {}).get("avatar_url", "")
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
                        seller_thumbnail=hit['_source'].get("seller", {}).get("avatar_url", "")
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
        "ratings_relevance_threshold": 0.66,  
        "ratings_boost_multiplier": 1     
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


@jit(nopython=True)
def wilson_lower_bound_jit(pos, n, z=1.96):
    """JIT-optimized Wilson Lower Bound calculation"""
    if n == 0:
        return 0.0
    pos_rate = pos / n
    inner_sqrt = max(0, pos_rate * (1 - pos_rate) + z*z / (4 * n))
    numerator = pos_rate + z*z / (2 * n) - z * math.sqrt(inner_sqrt / n)
    denominator = 1 + z*z / n
    return max(0, min(1, numerator / denominator))

@jit(nopython=True)
def calculate_wilson_ratings_boost_jit(ratings_count, ratings_score, max_ratings=500):
    # Your existing logic, simplified for Numba
    if ratings_count <= 0 or ratings_score < 0 or ratings_score > 5:
        return 0.0
    pos_ratings = min(round(ratings_score * ratings_count), ratings_count)
    wilson_score = wilson_lower_bound_jit(pos_ratings, ratings_count)
    log_factor = math.log(1 + ratings_count) / math.log(1 + max_ratings)
    return 0.7 * wilson_score + 0.3 * log_factor

@jit(nopython=True)
def normalize_scores_batch(scores, max_score):
    """Batch normalize scores with JIT"""
    normalized = np.zeros_like(scores)
    for i in range(len(scores)):
        norm = scores[i] / max_score if max_score > 0 else 0
        normalized[i] = 1 / (1 + np.exp(-10 * (norm - 0.5)))
    return normalized


class AsyncCachedEmbeddingService:
    def __init__(self, colbert_service):
        self.colbert_service = colbert_service
        self.cache = {}
        self.max_cache_size = 1000
        self._cache_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    async def get_query_embedding_async(self, query_text):
        """Get embedding with async caching logic"""
        # First check if the query is in cache
        async with self._cache_lock:
            if query_text in self.cache:
                return self.cache[query_text]
        
        # Use thread pool for CPU-intensive embedding computation
        loop = asyncio.get_running_loop()
        try:
            # Use the sentence embedding to get a single vector rather than token-level embeddings
            embedding_metadata = await loop.run_in_executor(
                self._executor,
                self.colbert_service.get_colbert_sentence_embedding,
                query_text
            )
            embedding = embedding_metadata.embedding.tolist()[0]
            
            # Update cache
            async with self._cache_lock:
                if len(self.cache) >= self.max_cache_size:
                    # Remove the oldest item (FIFO)
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                
                self.cache[query_text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Re-raise to allow proper error handling upstream
            raise


# Singleton factory function for the async cached embedding service
async def get_async_cached_embedding_service(colbert = Depends(get_colbert)):
    """
    Returns a singleton instance of AsyncCachedEmbeddingService.
    This ensures we have only one cache across all requests.
    """
    # Use app state to store the singleton if available
    if hasattr(get_async_cached_embedding_service, "instance"):
        return get_async_cached_embedding_service.instance
    
    # Create new instance if none exists
    instance = AsyncCachedEmbeddingService(colbert)
    get_async_cached_embedding_service.instance = instance
    
    return instance

@jit(nopython=True)
def batch_cosine_similarity(query_vector, doc_vectors):
    """
    Vectorized dot product for pre-normalized vectors.
    """
    # One line, fully vectorized dot product
    return np.dot(doc_vectors, query_vector)



async def get_embedding_async(query_text, embedding_service):
    """Async wrapper for embedding service with proper fallback mechanisms"""
    try:
        # Try using the async method if available
        if hasattr(embedding_service, 'get_query_embedding_async'):
            return await embedding_service.get_query_embedding_async(query_text)
        
        # Fallback to sync method using run_in_executor if available
        loop = asyncio.get_running_loop()
        if hasattr(embedding_service, 'get_query_embedding'):
            return await loop.run_in_executor(
                None,
                embedding_service.get_query_embedding,
                query_text
            )
        
        # Last resort fallback to direct ColBERT access
        if hasattr(embedding_service, 'colbert_service'):
            colbert_metadata = await loop.run_in_executor(
                None,
                embedding_service.colbert_service.get_colbert_sentence_embedding,
                query_text
            )
            return colbert_metadata.embedding.tolist()[0]
        
        # If we get here, we really have no way to get embeddings
        raise ValueError("No viable embedding method found on service")
        
    except Exception as e:
        logger.error(f"Error in get_embedding_async: {str(e)}")
        raise




@app.post("/two_phase_unnative_optimized", response_model=SearchResponse)
async def two_phase_unnative_optimized(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    colbert = Depends(get_colbert),
    embedding_service = Depends(get_async_cached_embedding_service)
):
    start_time = time.time()
    
    try:
        query_split = query.query.split()

        bm25_query = {
            "_source": ["name", "description", "thumbnail_url", "id", 
                      "ratings.count", "ratings.average", "price_cents", "url", 
                      "seller.id", "seller.name", "seller.avatar_url", "description_embedding"],
            "query": {
                "bool": {
                    "should": [
                        {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "AND", "boost": 1.5}},
                        {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "OR", "boost": 1.2}},
                        {"fuzzy": {"name": {"value": query.query, "fuzziness": get_fuzziness(query.query), "boost": 1}}}
                    ]
                }
            },
            "size": min(100, query.num_candidates)
        }

        # Execute BM25 search and embedding computation in parallel with timeout
        query_embedding_task = asyncio.create_task(get_embedding_async(query.query, embedding_service))
        bm25_response_task = asyncio.create_task(es_client.search(index='products', body=bm25_query))
        
        # Add a timeout to avoid hanging forever
        try:
            done, pending = await asyncio.wait(
                [query_embedding_task, bm25_response_task], 
                timeout=10.0,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Check if all tasks completed
            if len(done) < 2:
                raise TimeoutError("Some tasks did not complete in time")
            
            # Get results
            query_vector = await query_embedding_task
            bm25_response = await bm25_response_task
            
        except TimeoutError:
            logger.warning("Timeout occurred while waiting for tasks")
            # If BM25 completed but not embeddings, we can still return results
            if bm25_response_task.done() and not query_embedding_task.done():
                bm25_response = await bm25_response_task
                query_vector = None
            else:
                # If BM25 didn't complete, we can't proceed
                return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
        
        candidates = bm25_response['hits']['hits']
        
        if not candidates:
            return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
        
        # Get candidate IDs for processing
        candidate_ids = [hit['_id'] for hit in candidates]
        
        # Create mapping for document data
        doc_data = {hit['_id']: hit for hit in candidates}
        
        # Process document embeddings and calculate similarities directly
        similarity_scores = {}
        if query_vector is not None:
            query_vector_array = np.array(query_vector, dtype=np.float32)
            
            # Extract embeddings and calculate similarities in one pass
            for hit in candidates:
                doc_id = hit['_id']
                embedding = hit['_source'].get('description_embedding')
                if embedding:
                    # Calculate similarity directly
                    doc_vector = np.array(embedding, dtype=np.float32)
                    similarity_scores[doc_id] = float(np.dot(doc_vector, query_vector_array))
        
        # Determine if this is a multi-term query
        is_multi_term = len([t for t in query.query.split() if len(t) > 3 and t not in stopwords_set]) > 1

        bm25_weight = 0.35 if is_multi_term else 0.65
        colbert_weight = 1 - bm25_weight
        
        # Process results using simple, reliable code
        reranked_results = []
        for doc_id in candidate_ids:
            hit = doc_data[doc_id]
            
            # # Get ratings data
            ratings = hit['_source'].get('ratings', {})
            ratings_count = ratings.get('count', 0) if isinstance(ratings, dict) else 0
            ratings_score = ratings.get('average', -1.0) if isinstance(ratings, dict) else -1.0
            
            # Normalize BM25 score
            bm25_score = hit.get('_score', 0)
            normalized_bm25 = 1 - 2 / (1 + bm25_score)
            
            # Get colbert similarity score
            colbert_score = similarity_scores.get(doc_id, 0)

            combined_score = (normalized_bm25 * bm25_weight) + (colbert_score * colbert_weight)
            
            
            # Get seller data
            seller = hit['_source'].get('seller', {})
            seller_id = seller.get('id', '') if isinstance(seller, dict) else ''
            seller_name = seller.get('name', '') if isinstance(seller, dict) else ''
            seller_avatar = seller.get('avatar_url', '') if isinstance(seller, dict) else ''
            
            # Create result
            reranked_results.append(SearchResult(
                score=float(combined_score),
                base_score=float(normalized_bm25),
                other_score=float(colbert_score),
                score_origin="hybrid",
                name=hit['_source'].get('name', ''),
                description=hit['_source'].get('description', ''),
                thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                ratings_count=int(ratings_count),
                ratings_score=float(ratings_score),
                price_cents=hit['_source'].get('price_cents', 0),
                url=hit['_source'].get('url', ''),
                id=doc_id,
                seller_id=seller_id,
                seller_name=seller_name,
                seller_thumbnail=seller_avatar
            ))
        
        # Sort by score
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        counter = 0
        for x in reranked_results[::-1]:
            print(x.ratings_count, x.ratings_score, x.base_score, x.other_score, x.score, counter, x.name)
            counter += 1
        
        # Calculate query time
        query_time = (time.time() - start_time) * 1000
        
        return SearchResponse(results=reranked_results[:query.k], query_time_ms=query_time)
        
    except Exception as e:
        # Log the error and return an empty response
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
            


@app.post("/elasticsearch_vector_optimized", response_model=SearchResponse)
async def elasticsearch_vector_optimized(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    embedding_service = Depends(get_async_cached_embedding_service)
):
    start_time = time.time()
    
    try:
        # Get query embedding in parallel with BM25 search
        query_embedding_task = asyncio.create_task(get_embedding_async(query.query, embedding_service))
        
        # BM25 query component
        bm25_component = {
            "bool": {
                "should": [
                    {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "OR"}},
                    {"fuzzy": {"name": {"value": query.query, "fuzziness": get_fuzziness(query.query), "boost": 1.0}}}
                ]
            }
        }
        
        # First phase: Execute BM25 search to get initial candidates
        bm25_query = {
            "_source": ["name", "description", "thumbnail_url", "id", 
                      "ratings.count", "ratings.average", "price_cents", "url", 
                      "seller.id", "seller.name", "seller.avatar_url"],
            "query": bm25_component,
            "size": min(100, query.num_candidates)
        }
        
        bm25_response_task = asyncio.create_task(es_client.search(index='products', body=bm25_query))
        
        # Wait for both tasks with timeout
        try:
            done, pending = await asyncio.wait(
                [query_embedding_task, bm25_response_task], 
                timeout=10.0,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Check if all tasks completed
            if len(done) < 2:
                raise TimeoutError("Some tasks did not complete in time")
            
            # Get results
            query_vector = await query_embedding_task
            bm25_response = await bm25_response_task
            
        except TimeoutError:
            logger.warning("Timeout occurred while waiting for tasks")
            # If BM25 completed but not embeddings, we can still return results
            if bm25_response_task.done() and not query_embedding_task.done():
                bm25_response = await bm25_response_task
                query_vector = None
            else:
                # If BM25 didn't complete, we can't proceed
                return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
        
        # If no embedding was generated or no BM25 results, use BM25 results only
        if query_vector is None or not bm25_response['hits']['hits']:
            candidates = bm25_response['hits']['hits']
            
            if not candidates:
                return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
            
            # Process BM25 results only
            results = []
            for hit in candidates:
                # Get ratings data
                ratings = hit['_source'].get('ratings', {})
                ratings_count = ratings.get('count', 0) if isinstance(ratings, dict) else 0
                ratings_score = ratings.get('average', -1.0) if isinstance(ratings, dict) else -1.0
                
                # Normalize BM25 score
                bm25_score = hit.get('_score', 0)
                normalized_bm25 = 1 - 2 / (1 + bm25_score)
                
                # Get seller data
                seller = hit['_source'].get('seller', {})
                seller_id = seller.get('id', '') if isinstance(seller, dict) else ''
                seller_name = seller.get('name', '') if isinstance(seller, dict) else ''
                seller_avatar = seller.get('avatar_url', '') if isinstance(seller, dict) else ''
                
                results.append(SearchResult(
                    score=float(normalized_bm25),
                    base_score=float(normalized_bm25),
                    other_score=0.0,
                    score_origin="bm25",
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=int(ratings_count),
                    ratings_score=float(ratings_score),
                    price_cents=hit['_source'].get('price_cents', 0),
                    url=hit['_source'].get('url', ''),
                    id=hit['_id'],
                    seller_id=seller_id,
                    seller_name=seller_name,
                    seller_thumbnail=seller_avatar
                ))
            
            # Sort by score and return
            results.sort(key=lambda x: x.score, reverse=True)
            return SearchResponse(
                results=results[:query.k], 
                query_time_ms=(time.time() - start_time) * 1000
            )
        
        # If we have both embeddings and BM25 results, use a hybrid approach
        
        # Get the candidate IDs from BM25 for filtering
        candidate_ids = [hit['_id'] for hit in bm25_response['hits']['hits']]
        
        # Determine if this is a multi-term query
        is_multi_term = len([t for t in query.query.split() if len(t) > 3]) > 1
        bm25_weight = 0.4 if is_multi_term else 0.6
        vector_weight = 0.6 if is_multi_term else 0.4
        
        # Store BM25 scores for later use
        bm25_scores = {hit['_id']: hit.get('_score', 0) for hit in bm25_response['hits']['hits']}
        
        # Create and execute a hybrid query with vector search restricted to BM25 candidates
        hybrid_query = {
            "_source": ["name", "description", "thumbnail_url", "id", 
                      "ratings.count", "ratings.average", "price_cents", "url", 
                      "seller.id", "seller.name", "seller.avatar_url"],
            "query": {
                "bool": {
                    "must": [
                        {
                            "ids": {
                                "values": candidate_ids
                            }
                        },
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'description_embedding') + 1.0",
                                    "params": {
                                        "query_vector": query_vector
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "size": query.k
        }
        
        vector_response = await es_client.search(index='products', body=hybrid_query)
        
        # Process results with combined scoring
        results = []
        for hit in vector_response['hits']['hits']:
            doc_id = hit['_id']
            
            # Get ratings data
            ratings = hit['_source'].get('ratings', {})
            ratings_count = ratings.get('count', 0) if isinstance(ratings, dict) else 0
            ratings_score = ratings.get('average', -1.0) if isinstance(ratings, dict) else -1.0
            
            # Get and normalize BM25 score
            bm25_score = bm25_scores.get(doc_id, 0)
            normalized_bm25 = 1 - 2 / (1 + bm25_score)
            
            # Get vector score (already normalized by Elasticsearch)
            vector_score = hit.get('_score', 0) - 1.0  # Adjust for the +1.0 in the script
            
            # Combine scores
            combined_score = (normalized_bm25 * bm25_weight) + (vector_score * vector_weight)
            
            # Get seller data
            seller = hit['_source'].get('seller', {})
            seller_id = seller.get('id', '') if isinstance(seller, dict) else ''
            seller_name = seller.get('name', '') if isinstance(seller, dict) else ''
            seller_avatar = seller.get('avatar_url', '') if isinstance(seller, dict) else ''
            
            results.append(SearchResult(
                score=float(combined_score),
                base_score=float(normalized_bm25),
                other_score=float(vector_score),
                score_origin="hybrid",
                name=hit['_source'].get('name', ''),
                description=hit['_source'].get('description', ''),
                thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                ratings_count=int(ratings_count),
                ratings_score=float(ratings_score),
                price_cents=hit['_source'].get('price_cents', 0),
                url=hit['_source'].get('url', ''),
                id=doc_id,
                seller_id=seller_id,
                seller_name=seller_name,
                seller_thumbnail=seller_avatar
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return SearchResponse(
            results=results[:query.k], 
            query_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        # Log the error and return an empty response
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
    

@app.post("/two_phase_unnative_optimized", response_model=SearchResponse)
async def two_phase_unnative_optimized(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    colbert = Depends(get_colbert),
    embedding_service = Depends(get_async_cached_embedding_service)
):
    start_time = time.time()
    
    try:
        query_split = query.query.split()

        bm25_query = {
            "_source": ["name", "description", "thumbnail_url", "id", 
                      "ratings.count", "ratings.average", "price_cents", "url", 
                      "seller.id", "seller.name", "seller.avatar_url", "description_embedding"],
            "query": {
                "bool": {
                    "should": [
                        {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "AND", "boost": 1.5}},
                        {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "OR", "boost": 1.2}},
                        {"fuzzy": {"name": {"value": query.query, "fuzziness": get_fuzziness(query.query), "boost": 1}}}
                    ]
                }
            },
            "size": min(100, query.num_candidates)
        }

        # Execute BM25 search and embedding computation in parallel with timeout
        query_embedding_task = asyncio.create_task(get_embedding_async(query.query, embedding_service))
        bm25_response_task = asyncio.create_task(es_client.search(index='products', body=bm25_query))
        
        # Add a timeout to avoid hanging forever
        try:
            done, pending = await asyncio.wait(
                [query_embedding_task, bm25_response_task], 
                timeout=10.0,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Check if all tasks completed
            if len(done) < 2:
                raise TimeoutError("Some tasks did not complete in time")
            
            # Get results
            query_vector = await query_embedding_task
            bm25_response = await bm25_response_task
            
        except TimeoutError:
            logger.warning("Timeout occurred while waiting for tasks")
            # If BM25 completed but not embeddings, we can still return results
            if bm25_response_task.done() and not query_embedding_task.done():
                bm25_response = await bm25_response_task
                query_vector = None
            else:
                # If BM25 didn't complete, we can't proceed
                return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
        
        candidates = bm25_response['hits']['hits']
        
        if not candidates:
            return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
        
        # Get candidate IDs for processing
        candidate_ids = [hit['_id'] for hit in candidates]
        
        # Create mapping for document data
        doc_data = {hit['_id']: hit for hit in candidates}
        
        # Process document embeddings and calculate similarities directly
        similarity_scores = {}
        if query_vector is not None:
            query_vector_array = np.array(query_vector, dtype=np.float32)
            
            # Extract embeddings and calculate similarities in one pass
            for hit in candidates:
                doc_id = hit['_id']
                embedding = hit['_source'].get('description_embedding')
                if embedding:
                    # Calculate similarity directly
                    doc_vector = np.array(embedding, dtype=np.float32)
                    similarity_scores[doc_id] = float(np.dot(doc_vector, query_vector_array))
        
        # Determine if this is a multi-term query
        is_multi_term = len([t for t in query.query.split() if len(t) > 3 and t not in stopwords_set]) > 1

        bm25_weight = 0.35 if is_multi_term else 0.65
        colbert_weight = 1 - bm25_weight
        
        # Process results using simple, reliable code
        reranked_results = []
        for doc_id in candidate_ids:
            hit = doc_data[doc_id]
            
            # # Get ratings data
            ratings = hit['_source'].get('ratings', {})
            ratings_count = ratings.get('count', 0) if isinstance(ratings, dict) else 0
            ratings_score = ratings.get('average', -1.0) if isinstance(ratings, dict) else -1.0
            
            # Normalize BM25 score
            bm25_score = hit.get('_score', 0)
            normalized_bm25 = 1 - 2 / (1 + bm25_score)
            
            # Get colbert similarity score
            colbert_score = similarity_scores.get(doc_id, 0)

            combined_score = (normalized_bm25 * bm25_weight) + (colbert_score * colbert_weight)
            
            
            # Get seller data
            seller = hit['_source'].get('seller', {})
            seller_id = seller.get('id', '') if isinstance(seller, dict) else ''
            seller_name = seller.get('name', '') if isinstance(seller, dict) else ''
            seller_avatar = seller.get('avatar_url', '') if isinstance(seller, dict) else ''
            
            # Create result
            reranked_results.append(SearchResult(
                score=float(combined_score),
                base_score=float(normalized_bm25),
                other_score=float(colbert_score),
                score_origin="hybrid",
                name=hit['_source'].get('name', ''),
                description=hit['_source'].get('description', ''),
                thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                ratings_count=int(ratings_count),
                ratings_score=float(ratings_score),
                price_cents=hit['_source'].get('price_cents', 0),
                url=hit['_source'].get('url', ''),
                id=doc_id,
                seller_id=seller_id,
                seller_name=seller_name,
                seller_thumbnail=seller_avatar
            ))
        
        # Sort by score
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        counter = 0
        for x in reranked_results[::-1]:
            print(x.ratings_count, x.ratings_score, x.base_score, x.other_score, x.score, counter, x.name)
            counter += 1
        
        # Calculate query time
        query_time = (time.time() - start_time) * 1000
        
        return SearchResponse(results=reranked_results[:query.k], query_time_ms=query_time)
        
    except Exception as e:
        # Log the error and return an empty response
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
            
@app.post("/two_phase_optimized", response_model=SearchResponse)
async def two_phase_optimized(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip),
    colbert: ColBERTEmbedding = Depends(get_colbert)
):
    """
    Two-phase optimized search with vision fallback.
    - Uses BM25 for initial retrieval
    - Attempts to rerank BM25 results using ColBERT embeddings 
    - Falls back to CLIP vision search if needed
    """
    start_time = time.time()
    min_results = query.k  # Minimum number of results we want
    colbert_vector = None
    
    try:
        # PHASE 1: BM25 Search - Do this first before any embedding calculations
        bm25_query = {
            "_source": ["name", "description", "thumbnail_url", "id", 
                      "ratings", "price_cents", "url", "seller", "description_embedding"],
            "query": {
                "bool": {
                    "should": [
                        {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "AND", "boost": 1.5}},
                        {"combined_fields": {"query": query.query, "fields": ["name^3", "description"], "operator": "OR", "boost": 1.2}},
                        {"fuzzy": {"name": {"value": query.query, "fuzziness": get_fuzziness(query.query), "boost": 1}}}
                    ]
                }
            },
            "size": min(100, query.num_candidates)
        }
        
        # Execute BM25 search first - this is fast
        bm25_response = await es_client.search(index='products', body=bm25_query)
        candidates = bm25_response['hits']['hits']
        
        # If no BM25 results, use vision search directly
        if not candidates:
            return await vision_search_fallback(query, es_client, clip, start_time)
        
        # Try to get ColBERT embedding for reranking
        try:
            # Since we're in an async function, use a thread to avoid blocking
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                colbert_vector = await loop.run_in_executor(
                    executor, 
                    lambda: colbert.get_colbert_sentence_embedding(query.query).embedding.tolist()[0]
                )
        except Exception as e:
            logger.warning(f"Error getting ColBERT embedding: {str(e)}", exc_info=True)
            colbert_vector = None
        
        # Use the colbert vector we may have calculated
        query_vector = colbert_vector
        
        # Rerank BM25 results with hybrid scoring
        seen_ids = set()
        
        # Get candidate IDs for processing
        candidate_ids = [hit['_id'] for hit in candidates]
        
        # Create mapping for document data
        doc_data = {hit['_id']: hit for hit in candidates}
        
        # Process document embeddings and calculate similarities directly
        similarity_scores = {}
        if query_vector is not None:
            query_vector_array = np.array(query_vector, dtype=np.float32)
            
            # Extract embeddings and calculate similarities in one pass
            for hit in candidates:
                doc_id = hit['_id']
                embedding = hit['_source'].get('description_embedding')
                if embedding:
                    try:
                        # Calculate similarity directly
                        doc_vector = np.array(embedding, dtype=np.float32)
                        similarity_scores[doc_id] = float(np.dot(doc_vector, query_vector_array))
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for doc {doc_id}: {str(e)}")
                        similarity_scores[doc_id] = 0.0
        
        # Determine if this is a multi-term query
        is_multi_term = len([t for t in query.query.split() if len(t) > 3 and t not in stopwords_set]) > 1

        bm25_weight = 0.35 if is_multi_term else 0.65
        colbert_weight = 1 - bm25_weight
        
        # Process results using simple, reliable code
        reranked_results = []
        for doc_id in candidate_ids:
            hit = doc_data[doc_id]
            seen_ids.add(doc_id)
            
            # Get ratings data
            ratings = hit['_source'].get('ratings', {})
            ratings_count = ratings.get('count', 0) if isinstance(ratings, dict) else 0
            ratings_score = ratings.get('average', -1.0) if isinstance(ratings, dict) else -1.0
            
            # Normalize BM25 score
            bm25_score = hit.get('_score', 0)
            normalized_bm25 = 1 - 2 / (1 + bm25_score)
            
            # Get colbert similarity score
            colbert_score = similarity_scores.get(doc_id, 0)

            # If no colbert vector was calculated, only use BM25 score
            if query_vector is None:
                combined_score = normalized_bm25
            else:
                combined_score = (normalized_bm25 * bm25_weight) + (colbert_score * colbert_weight)
            
            # Get seller data
            seller = hit['_source'].get('seller', {})
            seller_id = seller.get('id', '') if isinstance(seller, dict) else ''
            seller_name = seller.get('name', '') if isinstance(seller, dict) else ''
            seller_avatar = seller.get('avatar_url', '') if isinstance(seller, dict) else ''
            
            # Create result
            reranked_results.append(SearchResult(
                score=float(combined_score),
                base_score=float(normalized_bm25),
                other_score=float(colbert_score) if query_vector is not None else 0.0,
                score_origin="hybrid" if query_vector is not None else "bm25",
                name=hit['_source'].get('name', ''),
                description=hit['_source'].get('description', ''),
                thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                ratings_count=int(ratings_count),
                ratings_score=float(ratings_score),
                price_cents=hit['_source'].get('price_cents', 0),
                url=hit['_source'].get('url', ''),
                id=doc_id,
                seller_id=seller_id,
                seller_name=seller_name,
                seller_thumbnail=seller_avatar
            ))
        
        # Sort by score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Check if we have enough results or need fallbacks
        if len(reranked_results) < min_results:
            # Not enough results, try vision search as fallback
            vision_results = await vision_search_fallback_with_exclusions(query, es_client, clip, seen_ids, start_time)
            
            # Combine results
            all_results = reranked_results + vision_results
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            query_time = (time.time() - start_time) * 1000
            return SearchResponse(results=all_results[:query.k], query_time_ms=query_time)
            
        # We have enough results, return them
        query_time = (time.time() - start_time) * 1000
        return SearchResponse(results=reranked_results[:query.k], query_time_ms=query_time)
        
    except Exception as e:
        # Log the error and return an empty response
        logger.error(f"Error in two_phase_optimized: {str(e)}", exc_info=True)
        return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)

async def vision_search_fallback(query, es_client, clip, start_time):
    """
    Perform vision search fallback - this is a simpler version that uses
    the proven search_vision functionality
    """
    try:
        # Calculate CLIP embedding
        embedding_result = clip.get_text_embedding(query.query)
        query_embedding = embedding_result.embedding
        
        # Perform search
        response = await es_client.search(
            index='products',
            body={
                "_source": [
                    "seller",
                    "description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"],
                "knn": {
                    "field": "image_embedding",
                    "query_vector": query_embedding.tolist()[0],
                    "k": query.k,
                    "num_candidates": query.num_candidates
                }
            }
        )
        
        # Process results
        results = []
        if response and 'hits' in response and 'hits' in response['hits']:
            for hit in response['hits']['hits']:
                # Get ratings data
                ratings_count = hit['_source'].get("ratings", {}).get("count", 0)
                ratings_score = hit['_source'].get("ratings", {}).get("average", -1.0)
                
                # Get seller data
                seller = hit['_source'].get('seller', {})
                seller_id = seller.get('id', '') if isinstance(seller, dict) else ''
                seller_name = seller.get('name', '') if isinstance(seller, dict) else ''
                seller_avatar = seller.get('avatar_url', '') if isinstance(seller, dict) else ''
                
                results.append(SearchResult(
                    score=hit['_score'],
                    base_score=hit['_score'],
                    other_score=0.0,
                    score_origin="clip",
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                    ratings_count=ratings_count,
                    ratings_score=ratings_score,
                    price_cents=hit['_source'].get("price_cents", 0),
                    url=hit['_source'].get("url", ""),
                    id=hit.get('_id', ''),
                    seller_id=seller_id,
                    seller_name=seller_name,
                    seller_thumbnail=seller_avatar
                ))
        
        query_time = (time.time() - start_time) * 1000
        return SearchResponse(results=results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Vision search fallback error: {str(e)}", exc_info=True)
        return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)

async def vision_search_fallback_with_exclusions(query, es_client, clip, exclusion_ids, start_time):
    """
    Perform vision search fallback but exclude IDs we've already seen
    """
    try:
        # Calculate CLIP embedding
        embedding_result = clip.get_text_embedding(query.query)
        query_embedding = embedding_result.embedding
        
        # Build search query with exclusions
        search_body = {
            "_source": [
                "seller", "description", "name", "thumbnail_url", "id", "ratings", "price_cents", "url"],
            "knn": {
                "field": "image_embedding",
                "query_vector": query_embedding.tolist()[0],
                "k": query.k * 2,  # Get more candidates since we'll be filtering
                "num_candidates": query.num_candidates * 2
            }
        }
        
        # Add exclusion filter if needed
        if exclusion_ids:
            search_body["query"] = {
                "bool": {
                    "must_not": [
                        {"ids": {"values": list(exclusion_ids)}}
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
                doc_id = hit.get('_id', '')
                if doc_id not in exclusion_ids:
                    # Get ratings data
                    ratings_count = hit['_source'].get("ratings", {}).get("count", 0)
                    ratings_score = hit['_source'].get("ratings", {}).get("average", -1.0)
                    
                    # Normalize score to be in the same range as BM25
                    clip_score = hit.get('_score', 0)
                    normalized_score = 0.8 * (1 - 2 / (1 + clip_score)) if clip_score > 0 else 0
                    
                    # Get seller data
                    seller = hit['_source'].get('seller', {})
                    seller_id = seller.get('id', '') if isinstance(seller, dict) else ''
                    seller_name = seller.get('name', '') if isinstance(seller, dict) else ''
                    seller_avatar = seller.get('avatar_url', '') if isinstance(seller, dict) else ''
                    
                    results.append(SearchResult(
                        score=normalized_score,
                        base_score=clip_score,
                        other_score=0.0,
                        score_origin="clip",
                        name=hit['_source'].get('name', ''),
                        description=hit['_source'].get('description', ''),
                        thumbnail_url=hit['_source'].get('thumbnail_url', ''),
                        ratings_count=ratings_count,
                        ratings_score=ratings_score,
                        price_cents=hit['_source'].get("price_cents", 0),
                        url=hit['_source'].get("url", ""),
                        id=doc_id,
                        seller_id=seller_id,
                        seller_name=seller_name,
                        seller_thumbnail=seller_avatar
                    ))
        
        return results
        
    except Exception as e:
        logger.error(f"Vision search fallback error: {str(e)}", exc_info=True)
        return []
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