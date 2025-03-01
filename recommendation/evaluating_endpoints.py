#!/usr/bin/env python3
import os
import json
import glob
import requests
from collections import defaultdict
import time

# Configuration
GUMROAD_DATA_DIR = "./gumroad_data"
SEARCH_DIR = "search_sample_pack"
API_ENDPOINT = "http://localhost:8000/search_text_based"

def load_gumroad_results(search_dir):
    """Load Gumroad search results from search directory."""
    product_files = glob.glob(os.path.join(GUMROAD_DATA_DIR, search_dir, "*_products_search*.json"))
    
    if not product_files:
        print(f"No product search files found in {search_dir}")
        
        # List all search directories
        print("Available search directories:")
        dirs = [d for d in os.listdir(GUMROAD_DATA_DIR) 
                if os.path.isdir(os.path.join(GUMROAD_DATA_DIR, d)) and d.startswith("search_")]
        for d in dirs:
            print(f"- {d}")
            
        return []
    
    # Sort files by name
    product_files.sort()
    
    # Take the first file
    first_file = product_files[0]
    
    try:
        with open(first_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            products = data.get('products', [])
            print(f"Loaded {len(products)} products from {first_file}")
            return products
    except Exception as e:
        print(f"Error loading {first_file}: {e}")
        return []

def query_api(endpoint, query, k=10):
    """Query the search API."""
    payload = {
        "query": query,
        "k": k,
        "num_candidates": 100
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        start_time = time.time()
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        query_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"API returned {len(results)} results in {query_time:.2f}s")
            return results
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return []

def extract_query_from_dirname(dirname):
    """Extract query string from directory name."""
    if dirname.startswith("search_"):
        return dirname[7:].replace("_", " ")
    return dirname

def calculate_similarity(str1, str2):
    """Calculate similarity between two strings."""
    # Convert to word sets
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    if not set1 or not set2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def find_matches(gumroad_results, api_results, threshold=0.5):
    """Find matches between Gumroad and API results."""
    matches = []
    
    for g_result in gumroad_results:
        g_name = g_result.get('name', '').lower()
        if not g_name:
            continue
        
        for a_result in api_results:
            a_name = a_result.get('name', '').lower()
            if not a_name:
                continue
            
            # Check for exact match
            if g_name == a_name:
                matches.append(('exact', g_name, a_name))
                continue
            
            # Check for substring match
            if g_name in a_name or a_name in g_name:
                matches.append(('substring', g_name, a_name))
                continue
            
            # Check for similarity
            similarity = calculate_similarity(g_name, a_name)
            if similarity >= threshold:
                matches.append(('similar', g_name, a_name, similarity))
    
    return matches

def analyze_top_results():
    """Analyze several search queries and compare results."""
    # Find search directories
    search_dirs = [d for d in os.listdir(GUMROAD_DATA_DIR) 
                  if os.path.isdir(os.path.join(GUMROAD_DATA_DIR, d)) and d.startswith("search_")]
    
    # Take first 10 directories (or fewer if there are fewer)
    search_dirs = search_dirs[:10]
    
    results = {}
    
    for search_dir in search_dirs:
        query = extract_query_from_dirname(search_dir)
        print(f"\n=== Analyzing query: '{query}' ===")
        
        # Load Gumroad results
        gumroad_results = load_gumroad_results(search_dir)
        if not gumroad_results:
            continue
        
        # Print top Gumroad results
        print("\nTop 3 Gumroad results:")
        for i, result in enumerate(gumroad_results[:3]):
            name = result.get('name', 'N/A')
            native_type = result.get('native_type', 'N/A')
            description = result.get('description', 'N/A')
            if description and len(description) > 50:
                description = description[:47] + "..."
            print(f"{i+1}. {name} | Type: {native_type}")
            print(f"   {description}")
        
        # Query API
        api_results = query_api(API_ENDPOINT, query)
        if not api_results:
            continue
        
        # Print top API results
        print("\nTop 3 API results:")
        for i, result in enumerate(api_results[:3]):
            name = result.get('name', 'N/A')
            score = result.get('score', 0)
            description = result.get('description', 'N/A')
            if description and len(description) > 50:
                description = description[:47] + "..."
            print(f"{i+1}. {name} | Score: {score:.2f}")
            print(f"   {description}")
        
        # Find matches
        matches = find_matches(gumroad_results, api_results)
        
        # Print match statistics
        print(f"\nFound {len(matches)} matches:")
        match_types = defaultdict(int)
        for match in matches:
            match_types[match[0]] += 1
        
        for match_type, count in match_types.items():
            print(f"- {match_type}: {count}")
        
        if matches:
            print("\nSample matches:")
            for i, match in enumerate(matches[:5]):
                if match[0] == 'similar':
                    print(f"{i+1}. {match[0]}: '{match[1]}' <-> '{match[2]}' (sim: {match[3]:.2f})")
                else:
                    print(f"{i+1}. {match[0]}: '{match[1]}' <-> '{match[2]}'")
        
        # Store results
        results[query] = {
            'gumroad_count': len(gumroad_results),
            'api_count': len(api_results),
            'matches': len(matches),
            'match_types': dict(match_types)
        }
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Analyzed {len(results)} queries")
    
    avg_match_rate = sum(r['matches'] / r['gumroad_count'] for q, r in results.items()) / len(results) if results else 0
    print(f"Average match rate: {avg_match_rate:.2%}")
    
    # Print match rates for each query
    print("\nMatch rates by query:")
    for query, result in results.items():
        match_rate = result['matches'] / result['gumroad_count'] if result['gumroad_count'] > 0 else 0
        print(f"- {query}: {match_rate:.2%} ({result['matches']} of {result['gumroad_count']})")

def analyze_sample_pack_query():
    """Deep dive into the sample pack query."""
    search_dir = "search_sample_pack"
    query = "sample pack"
    
    print(f"\n=== Deep dive into '{query}' query ===")
    
    # Load Gumroad results
    gumroad_results = load_gumroad_results(search_dir)
    if not gumroad_results:
        return
    
    # Print all Gumroad results
    print("\nAll Gumroad results:")
    for i, result in enumerate(gumroad_results):
        name = result.get('name', 'N/A')
        native_type = result.get('native_type', 'N/A')
        print(f"{i+1}. {name} | Type: {native_type}")
    
    # Try multiple endpoints
    endpoints = {
        "text_based": "http://localhost:8000/search_text_based",
        "fuzzy": "http://localhost:8000/search_fuzzy",
        "vision": "http://localhost:8000/search_vision",
        "combined": "http://localhost:8000/search_combined",
        "lame_combined": "http://localhost:8000/search_lame_combined"
    }
    
    best_matches = 0
    best_endpoint = None
    
    for name, endpoint in endpoints.items():
        print(f"\nTrying endpoint: {name}")
        api_results = query_api(endpoint, query)
        
        if not api_results:
            continue
        
        matches = find_matches(gumroad_results, api_results)
        
        if len(matches) > best_matches:
            best_matches = len(matches)
            best_endpoint = name
        
        # Print match statistics
        print(f"Found {len(matches)} matches")
        
        if matches:
            print("Sample matches:")
            for i, match in enumerate(matches[:3]):
                if match[0] == 'similar':
                    print(f"{i+1}. {match[0]}: '{match[1]}' <-> '{match[2]}' (sim: {match[3]:.2f})")
                else:
                    print(f"{i+1}. {match[0]}: '{match[1]}' <-> '{match[2]}'")
    
    print(f"\nBest endpoint for '{query}': {best_endpoint} with {best_matches} matches")
    
    # Try variation of the query
    variations = [
        "sample pack",
        "sample packs",
        "samples pack",
        "samples",
        "audio samples",
        "music samples",
        "drum samples",
        "sound samples"
    ]
    
    best_var_matches = 0
    best_variation = None
    best_var_endpoint = None
    
    print("\nTrying query variations:")
    
    for variation in variations:
        print(f"\nQuery variation: '{variation}'")
        
        for name, endpoint in endpoints.items():
            api_results = query_api(endpoint, variation)
            
            if not api_results:
                continue
            
            matches = find_matches(gumroad_results, api_results)
            
            print(f"- {name}: {len(matches)} matches")
            
            if len(matches) > best_var_matches:
                best_var_matches = len(matches)
                best_variation = variation
                best_var_endpoint = name
    
    print(f"\nBest query variation: '{best_variation}' on {best_var_endpoint} with {best_var_matches} matches")

if __name__ == "__main__":
    print("=== Search Results Analysis Tool ===")
    
    # Check if API is accessible
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API is accessible")
            print(f"Health status: {health_response.json()}")
        else:
            print(f"⚠️ API returned error: {health_response.status_code}")
    except Exception as e:
        print(f"⚠️ Could not connect to API: {e}")
    
    # Analyze the sample pack query specifically
    analyze_sample_pack_query()
    
    # Analyze more queries for comparison
    analyze_top_results()