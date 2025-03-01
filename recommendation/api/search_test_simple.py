import requests

# API endpoint 
url = "http://localhost:8000/search_combined_simplified_but_slow"

# Example query payload
payload = {
    "query": "poli",  # Your search query
    "k": 10,                             # Number of results to return
    "num_candidates": 50                # Number of candidates to consider
}

# Headers for JSON content
headers = {"Content-Type": "application/json"}

# Send POST request
response = requests.post(url, json=payload, headers=headers)

# Print the results
print(response.json())

# Alternatively, you can print results more readably
results = response.json()
print("\nSearch Results:")
for result in results.get('results', []):
    print(f"Name: {result['name']}")
    print(f"Description: {result['description']}")
    print(f"Thumbnail URL: {result['thumbnail_url']}")
    print(f"Score: {result['score']}")
    print("---")
print(f"\nQuery Time: {results.get('query_time_ms', 0):.2f} ms")