import math
from collections import defaultdict

def rank_sellers(search_results):
    """
    Rank sellers based on their items' relevance scores in search results.
    
    Parameters:
    search_results: List of dictionaries, each containing at least:
                   - 'seller_id': ID of the seller
                   - 'relevance': Relevance score of the item
    
    Returns:
    List of tuples (seller_id, score) sorted by score in descending order
    """
    # Initialize a dictionary to store cumulative scores for each seller
    seller_scores = defaultdict(float)
    
    # Calculate discounted relevance scores for each item and accumulate by seller
    for position, item in enumerate(search_results):
        seller_id = item['seller_id']
        relevance = item['relevance']
        
        # Apply logarithmic discount based on position (similar to NDCG)
        discount = math.log2(position + 2)  # +2 because log₂(1) = 0
        discounted_score = relevance / discount
        
        # Add to seller's cumulative score
        seller_scores[seller_id] += discounted_score
    
    # Convert to list of tuples and sort by score in descending order
    ranked_sellers = sorted(seller_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_sellers

# Example usage
if __name__ == "__main__":
    # Sample search results (position is implicit in the list order)
    sample_results = [
        {'item_id': 'item1', 'seller_id': 'seller_A', 'relevance': 0.95},
        {'item_id': 'item2', 'seller_id': 'seller_B', 'relevance': 0.90},
        {'item_id': 'item3', 'seller_id': 'seller_A', 'relevance': 0.85},
        {'item_id': 'item4', 'seller_id': 'seller_C', 'relevance': 0.80},
        {'item_id': 'item5', 'seller_id': 'seller_B', 'relevance': 0.75},
        {'item_id': 'item6', 'seller_id': 'seller_D', 'relevance': 0.70},
        {'item_id': 'item7', 'seller_id': 'seller_A', 'relevance': 0.65},
        {'item_id': 'item8', 'seller_id': 'seller_C', 'relevance': 0.60},
        {'item_id': 'item9', 'seller_id': 'seller_E', 'relevance': 0.55},
        {'item_id': 'item10', 'seller_id': 'seller_B', 'relevance': 0.50},
    ]
    
    ranked_sellers = rank_sellers(sample_results)
    
    print("Seller Rankings:")
    for rank, (seller, score) in enumerate(ranked_sellers, 1):
        print(f"{rank}. Seller {seller}: {score:.4f}")
    
    # Calculate seller presence metrics
    seller_items_count = defaultdict(int)
    for item in sample_results:
        seller_items_count[item['seller_id']] += 1
    
    print("\nSeller Item Counts:")
    for seller, count in sorted(seller_items_count.items(), key=lambda x: x[1], reverse=True):
        print(f"Seller {seller}: {count} items")
    
    # Alternative approach: Average position of items for each seller
    seller_positions = defaultdict(list)
    for pos, item in enumerate(sample_results, 1):
        seller_positions[item['seller_id']].append(pos)
    
    print("\nAverage Position by Seller:")
    for seller, positions in seller_positions.items():
        avg_position = sum(positions) / len(positions)
        print(f"Seller {seller}: {avg_position:.2f} (positions: {positions})")

# Extended version with more sophisticated metrics

def advanced_seller_ranking(search_results, top_k=None, position_weight=1.0, relevance_weight=1.0):
    """
    More advanced seller ranking with configurable parameters.
    
    Parameters:
    search_results: List of dictionaries with search results
    top_k: Only consider top K results (None = use all)
    position_weight: Weight for position-based scoring (higher = position matters more)
    relevance_weight: Weight for relevance-based scoring (higher = relevance matters more)
    
    Returns:
    Dictionary with various seller metrics and rankings
    """
    if top_k is not None:
        search_results = search_results[:top_k]
    
    # Initialize tracking dictionaries
    seller_scores = defaultdict(float)
    seller_items = defaultdict(list)
    seller_relevance = defaultdict(list)
    seller_positions = defaultdict(list)
    
    # Process search results
    for position, item in enumerate(search_results, 1):
        seller_id = item['seller_id']
        relevance = item['relevance']
        
        # Track item details by seller
        seller_items[seller_id].append(item['item_id'])
        seller_relevance[seller_id].append(relevance)
        seller_positions[seller_id].append(position)
        
        # Calculate position-discounted relevance (NDCG-inspired)
        position_discount = math.log2(position + 1)
        discounted_score = (relevance * relevance_weight) / (position_discount ** position_weight)
        seller_scores[seller_id] += discounted_score
    
    # Calculate additional metrics
    seller_metrics = {}
    for seller_id in seller_items.keys():
        num_items = len(seller_items[seller_id])
        avg_relevance = sum(seller_relevance[seller_id]) / num_items
        avg_position = sum(seller_positions[seller_id]) / num_items
        best_position = min(seller_positions[seller_id])
        best_relevance = max(seller_relevance[seller_id])
        
        seller_metrics[seller_id] = {
            'score': seller_scores[seller_id],
            'item_count': num_items,
            'items': seller_items[seller_id],
            'avg_relevance': avg_relevance,
            'avg_position': avg_position,
            'best_position': best_position,
            'best_relevance': best_relevance,
            'relevance_list': seller_relevance[seller_id],
            'position_list': seller_positions[seller_id]
        }
    
    # Sort sellers by score
    ranked_sellers = sorted(seller_metrics.items(), key=lambda x: x[1]['score'], reverse=True)
    
    return {
        'ranked_sellers': ranked_sellers,
        'metrics': seller_metrics
    }