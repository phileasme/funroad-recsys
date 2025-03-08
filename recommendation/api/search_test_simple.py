import requests


# // Create default data for all profiles
# const searchProfiles = [
#   { id: 'search_text_based', name: 'Text-Based Search' },
#   { id: 'search_fuzzy', name: 'Fuzzy Search' },
#   { id: 'search_vision', name: 'Vision Search' },
#   { id: 'search_colbert', name: 'ColBERT Search' },
#   { id: 'search_combined', name: 'Combined Search' },
#   { id: 'search_lame_combined', name: 'Basic Combined' },
#   { id: 'search_combined_simplified_but_slow', name: ' Combined No rating', version: "(v0.7)" },
#   {id: 'search_combined_v0_8', name: 'Combine Perf Optimized', version: "(v0.8)"}
# ];

# API endpoint 
# url = "http://localhost:8000/similar_vision"
# search_combined_simplified_but_slow
# url = "http://localhost:8000/unn"
url = "http://localhost:8000/search_combined_v0_8"

# Example query payload
payload = {
    "query": "poker", 
    "k": 10,                             
    "num_candidates": 50,
    # "id": "3hrx2jgIR1pDpJA-nM11xw=="

}

# Headers for JSON content
headers = {"Content-Type": "application/json"}

# Send POST request
response = requests.post(url, json=payload, headers=headers)

# Alternatively, you can print results more readably
results = response.json()
print("\nSearch Results:")
for result in results.get('results', []):
    for fields, values in result.items():
        print(fields, values)
print(f"\nQuery Time: {results.get('query_time_ms', 0):.2f} ms")