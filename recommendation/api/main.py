from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from elasticsearch import AsyncElasticsearch as Elasticsearch
import logging
import os
from contextlib import asynccontextmanager
import numpy as np
import torch
import time
import pathlib
import sys

import asyncio
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
    if await app_state.es_client:
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


# Mount static files directory
static_dir = pathlib.Path(__file__).parent.parent / "static"
if not static_dir.exists():
    logger.info(f"Creating static directory: {static_dir}")
    static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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

# Root endpoint - serve the search UI
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = static_dir / "index.html"
    logger.info(f"Looking for HTML file at: {html_file}")
    if html_file.exists():
        with open(html_file, "r") as f:
            return f.read()
    else:
        logger.warning(f"HTML file not found at: {html_file}")
        return RedirectResponse(url="/docs")

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
                "_source": ["description", "name", "thumbnail_url"],
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
                results.append(SearchResult(
                    score=hit['_score'],
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
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
                "_source": ["description", "name", "thumbnail_url"],
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
                "_source": ["description", "name", "thumbnail_url"],
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
                "_source": ["description", "name", "thumbnail_url"],
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
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
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
                    "_source": ["description", "name", "thumbnail_url"],
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
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
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
                "_source": ["description", "name", "thumbnail_url"],
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
                    "_source": ["description", "name", "thumbnail_url"],
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
                        name=source.get('name', ''),
                        description=source.get('description', ''),
                        thumbnail_url=source.get('thumbnail_url', '')
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

def get_fuzziness(query_text):
    length = len(query_text)
    if length <= 3:
        return 0
    elif length <= 8:
        return 1
    else:
        return 2
    
def normalize_score(score, max_score, min_val=0.3, max_val=1.0):
    if max_score == 0:
        return min_val
    return min(max_val, max(min_val, score / max_score))
    

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
            "_source": ["description", "name", "thumbnail_url", "id"],
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
            "_source": ["description", "name", "thumbnail_url", "id"],
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
            "_source": ["description", "name", "thumbnail_url", "id"],
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
        results.append(SearchResult(score=score, **hit['_source']))
    
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
            results.append(SearchResult(score=score, **hit['_source']))
    
    # Phase 3: CLIP Search (only if needed)
    if not results or results[0].score < min_acceptable_score:
        clip_results = await clip_search()
        for hit in clip_results:
            doc_id = hit['_id']
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                score = min(0.6, max(0.2, hit['_score']))
                results.append(SearchResult(score=score, **hit['_source']))
    
    results.sort(key=lambda x: x.score, reverse=True)
    return SearchResponse(results=results[:query.k], query_time_ms=(time.time() - start_time) * 1000)


@app.post("/search_combined_simplified_but_slow", response_model=SearchResponse)
async def search_combined_optimal(
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
        "_source": ["description", "name", "thumbnail_url", "id"],
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
            name=hit['_source'].get('name', ''),
            description=hit['_source'].get('description', ''),
            thumbnail_url=hit['_source'].get('thumbnail_url', '')
        ))

    # ColBERT Semantic Search
    colbert_vector = colbert.get_colbert_sentence_embedding(query.query).embedding.tolist()[0]
    colbert_query = {"_source": ["description", "name", "thumbnail_url", "id"], "knn": {"field": "description_embedding", "query_vector": colbert_vector, "k": query.k, "num_candidates": query.num_candidates}}
    colbert_response = await es_client.search(index='products', body=colbert_query)
    colbert_hits = colbert_response['hits']['hits']
    
    for hit in colbert_hits:
        doc_id = hit.get('_id')
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            results.append(SearchResult(
                score=normalize_score(hit.get('_score', 0), 1.0),
                name=hit['_source'].get('name', ''),
                description=hit['_source'].get('description', ''),
                thumbnail_url=hit['_source'].get('thumbnail_url', '')
            ))
    
    # CLIP Vision Search (If Needed)
    if not results or results[0].score < min_acceptable_score:
        clip_vector = await asyncio.wait_for(clip_future, timeout=0.5)
        clip_query = {"_source": ["description", "name", "thumbnail_url", "id"], "knn": {"field": "image_embedding", "query_vector": clip_vector, "k": query.k, "num_candidates": query.num_candidates}}
        clip_response = await es_client.search(index='products', body=clip_query)
        
        for hit in clip_response['hits']['hits']:
            doc_id = hit.get('_id')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append(SearchResult(
                    score=normalize_score(hit.get('_score', 0), 0.6),
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
                ))
    
    results.sort(key=lambda x: x.score, reverse=True)
    query_time = (time.time() - start_time) * 1000
    return SearchResponse(results=results[:query.k], query_time_ms=query_time)

@app.post("/search_combined_optimal", response_model=SearchResponse)
async def search_combined_optimal(
    query: SearchQuery,
    es_client: Elasticsearch = Depends(get_es_client),
    clip: CLIPEmbedding = Depends(get_clip),
    colbert: ColBERTEmbedding = Depends(get_colbert)
):
    """
    Simplified search approach with two phases:
    1. Phase 1: Text search + fuzzy match (distance 1) + ColBERT
    2. Phase 2: CLIP (only if needed)
    """
    start_time = time.time()
    min_acceptable_score = 0.4  # Threshold for considering results good enough
    results = []
    
    # Start CLIP embedding computation in background immediately
    # (We'll only use it if needed, but start computing now)
    executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_event_loop()
    clip_future = loop.run_in_executor(
        executor, 
        lambda: clip.get_text_embedding(query.query).embedding.tolist()[0]
    )
    
    try:
        # Phase 1: Text search + fuzzy match + ColBERT
        seen_ids = set()
        
        # 1.1: Text search with combined fields
        combined_response = await es_client.search(
            index='products',
            body={
                "_source": ["description", "name", "thumbnail_url", "id"],
                "query": {
                    "combined_fields": {
                        "query": query.query,
                        "fields": ["name^3", "description"],
                        "operator": "OR",
                        "auto_generate_synonyms_phrase_query": True
                    }
                },
                "size": query.k,
                "track_scores": True
            }
        )
        
        # Track text search hits
        text_hits = combined_response['hits']['hits']
        text_max_score = max([hit.get('_score', 0) for hit in text_hits]) if text_hits else 1.0
        
        # Process text search results
        for hit in text_hits:
            doc_id = hit.get('_id')
            if doc_id:
                seen_ids.add(doc_id)
                
                # Calculate score - direct text matches get high scores
                raw_score = hit.get('_score', 0)
                normalized_score = min(1.0, raw_score / (text_max_score * 1.2))
                
                # Boost exact matches
                if query.query.lower() in hit['_source'].get('name', '').lower():
                    normalized_score = min(1.0, normalized_score * 1.5)
                
                results.append(SearchResult(
                    score=normalized_score,
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
                ))
        
        # 1.2: Fuzzy search with distance 1
        fuzzy_response = await es_client.search(
            index='products',
            body={
                "_source": ["description", "name", "thumbnail_url", "id"],
                "query": {
                    "bool": {
                        "should": [
                            {
                                "fuzzy": {
                                    "name": {
                                        "value": query.query,
                                        "fuzziness": 1,  # Exactly distance 1
                                        "prefix_length": 1,
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "fuzzy": {
                                    "description": {
                                        "value": query.query,
                                        "fuzziness": 1,  # Exactly distance 1
                                        "prefix_length": 1
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": query.k
            }
        )
        
        # Track fuzzy hits
        fuzzy_hits = fuzzy_response['hits']['hits']
        fuzzy_max_score = max([hit.get('_score', 0) for hit in fuzzy_hits]) if fuzzy_hits else 1.0
        
        # Process fuzzy search results
        for hit in fuzzy_hits:
            doc_id = hit.get('_id')
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                
                # Calculate score - fuzzy matches get good but slightly lower scores
                raw_score = hit.get('_score', 0)
                normalized_score = min(0.85, raw_score / (fuzzy_max_score * 1.5))
                
                results.append(SearchResult(
                    score=normalized_score,
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
                ))
        
        # 1.3: ColBERT semantic search
        colbert_embedding = colbert.get_colbert_sentence_embedding(query.query)
        colbert_vector = colbert_embedding.embedding.tolist()[0]
        
        colbert_response = await es_client.search(
            index='products',
            body={
                "_source": ["description", "name", "thumbnail_url", "id"],
                "knn": {
                    "field": "description_embedding",
                    "query_vector": colbert_vector,
                    "k": query.k,
                    "num_candidates": query.num_candidates
                }
            }
        )
        
        # Track colbert hits
        colbert_hits = colbert_response['hits']['hits']
        
        # Process ColBERT search results
        for hit in colbert_hits:
            doc_id = hit.get('_id')
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                
                # Calculate score - semantic matches get moderate scores
                raw_score = hit.get('_score', 0)
                normalized_score = min(0.7, max(0.3, raw_score))
                
                # Boost if it contains query terms despite being semantic search
                if query.query.lower() in hit['_source'].get('name', '').lower() or \
                   query.query.lower() in hit['_source'].get('description', '').lower():
                    normalized_score = min(0.8, normalized_score * 1.3)
                
                results.append(SearchResult(
                    score=normalized_score,
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
                ))
        
        # Sort results and check if we have good enough scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Phase 2: Only use CLIP if we don't have good enough results
        if not results or results[0].score < min_acceptable_score:
            logger.info("Phase 1 results insufficient, trying CLIP")
            
            # Get CLIP vector from background task
            try:
                clip_vector = await asyncio.wait_for(clip_future, timeout=0.5)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Background CLIP computation failed: {e}, computing now")
                clip_text_embedding = clip.get_text_embedding(query.query)
                clip_vector = clip_text_embedding.embedding.tolist()[0]
            
            clip_response = await es_client.search(
                index='products',
                body={
                    "_source": ["description", "name", "thumbnail_url", "id"],
                    "knn": {
                        "field": "image_embedding",
                        "query_vector": clip_vector,
                        "k": query.k,
                        "num_candidates": query.num_candidates
                    }
                }
            )
            
            # Process CLIP results
            for hit in clip_response['hits']['hits']:
                doc_id = hit.get('_id')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    
                    # Calculate score - visual matches get lower base scores
                    raw_score = hit.get('_score', 0)
                    normalized_score = min(0.6, max(0.2, raw_score))
                    
                    # Boost if it contains query terms despite being visual search
                    if query.query.lower() in hit['_source'].get('name', '').lower() or \
                       query.query.lower() in hit['_source'].get('description', '').lower():
                        normalized_score = min(0.75, normalized_score * 1.5)
                    
                    results.append(SearchResult(
                        score=normalized_score,
                        name=hit['_source'].get('name', ''),
                        description=hit['_source'].get('description', ''),
                        thumbnail_url=hit['_source'].get('thumbnail_url', '')
                    ))
            
            # Final sort after adding CLIP results
            results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to requested size
        results = results[:query.k]
        
        query_time = (time.time() - start_time) * 1000
        return SearchResponse(results=results, query_time_ms=query_time)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        
        # Simple fallback search
        try:
            quick_response = await es_client.search(
                index='products',
                body={
                    "_source": ["description", "name", "thumbnail_url"],
                    "query": {
                        "match": {
                            "name": query.query
                        }
                    },
                    "size": query.k
                }
            )
            
            quick_results = []
            for hit in quick_response['hits']['hits']:
                quick_results.append(SearchResult(
                    score=0.5,  # Default score for fallback results
                    name=hit['_source'].get('name', ''),
                    description=hit['_source'].get('description', ''),
                    thumbnail_url=hit['_source'].get('thumbnail_url', '')
                ))
            
            query_time = (time.time() - start_time) * 1000
            return SearchResponse(results=quick_results, query_time_ms=query_time)
            
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {fallback_error}")
            return SearchResponse(results=[], query_time_ms=(time.time() - start_time) * 1000)
        
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
                "_source": ["description", "name", "thumbnail_url"],
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
                thumbnail_url=hit['_source'].get('thumbnail_url', '')
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