import requests


# // Create default data for all profiles
# const searchProfiles = [
#   { id: 'search_text_based', name: 'Text-Based Search' },
#   { id: 'search_fuzzy', name: 'Fuzzy Search' },
#   { id: 'search_vision', name: 'Vision Search' },
#   { id: 'search_colbert', name: 'ColBERT Search' },
#   { id: 'search_combined', name: 'Combined Search' },
#   { id: 'search_combined_simplified_but_slow', name: 'Optimized Combined' },
#   { id: 'search_lame_combined', name: 'Basic Combined' }
# ];

# API endpoint 
url = "http://localhost:8000/search_lame_combined"
# url = "http://localhost:8000/search_fuzzy"

# Example query payload
payload = {
    "query": "prumker",  # Your search query
    "k": 10,                             # Number of results to return
    "num_candidates": 50                # Number of candidates to consider
}

# Headers for JSON content
headers = {"Content-Type": "application/json"}

# Send POST request
response = requests.post(url, json=payload, headers=headers)

# Print the results

# Alternatively, you can print results more readably
results = response.json()
print("\nSearch Results:")
for result in results.get('results', []):
    for fields, values in result.items():
        print(fields, values)
#     # print(f"Name: {result['name']}")
#     # print(f"Description: {result['description']}")
#     # print(f"Thumbnail URL: {result['thumbnail_url']}")
#     # print(f"Score: {result['score']}")
#     print("---")
    # break
print(f"\nQuery Time: {results.get('query_time_ms', 0):.2f} ms")