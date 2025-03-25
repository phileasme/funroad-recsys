# Search & Discovery Case Study for Gumroad (TL&DR)
### Full report can be found here: [Notion Case Study](https://www.notion.so/Search-Discovery-Case-Study-Blog-40e476a45ad94596ad323289eac62c2c)
<img width="1439" alt="Screenshot 2025-03-25 at 14 37 16" src="https://github.com/user-attachments/assets/ecfae368-bd05-400e-928a-f97277560368" />


## Executive Summary
This case study analyzes Gumroad's current search functionality and provides recommendations for improvements across search relevance and product discovery. It includes both analysis of current issues and proposed solutions ranging from quick wins to more sophisticated ML-based approaches.

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
   - [Prototype and Experimentation](#prototype-and-experimentation)
   - [Issues Identified](#issues-identified)
      - [Title Weighting Issues](#title-weighting-issues)
      - [Lack of Fuzzy Matching](#lack-of-fuzzy-matching)
      - [Language Localization Gaps](#language-localization-gaps)
      - [Query Expansion Limitations](#query-expansion-limitations)
      - [User Experience Pain Points](#user-experience-pain-points)
2. [What Has Been Done](#what-has-been-done)
   - [Prototype Search Implementation](#prototype-search-implementation)
   - [Current Model Architecture](#current-model-architecture)
   - [Evaluation Metrics](#evaluation-metrics)
3. [Proposed Improvements](#proposed-improvements)
   - [Quick Wins (Low Effort, High Impact)](#quick-wins)
   - [Mid-term Improvements](#mid-term-improvements)
   - [Advanced Solutions](#advanced-solutions)
4. [Technical Implementation Details](#technical-implementation-details)
   - [Search and Retrieval Techniques](#search-and-retrieval-techniques)
   - [Recommendation Systems](#recommendation-systems)
   - [Personalization Approaches](#personalization-approaches)
5. [Supporting Notes](#supporting-notes)
   - [Data Engineering Considerations](#data-engineering-considerations)
   - [Deployment Considerations](#deployment-considerations)
   - [Experimental Design](#experimental-design)

---

## Current State Analysis

### Prototype and Experimentation
To assess the current state of Gumroad's search, approximately 5,000 products were sourced from the existing search API to build a prototype for evaluation. The prototype allowed for direct comparison with the current search experience.

### Issues Identified

#### Title Weighting Issues
The product title field appears to be underweighted in the current search algorithm. For example, when searching for "poker," the product "The Cloud Resume Challenge Guidebook" appears before "Poker Without Fear 2025!" despite the latter being more relevant to the query, having a higher rating, and more sales.

#### Lack of Fuzzy Matching
The current search engine fails to match queries with minor misspellings. Missing a query by even one character (e.g., "pker" instead of "poker") results in no relevant items being returned, creating a poor user experience.

#### Language Localization Gaps
The search experience doesn't account for the user's language preferences. Recommendations remain the same regardless of the browser language, query language, user history, or geographic region.

#### Query Expansion Limitations
When searching for general terms like "sound," results fail to include relevant products from the "Music & Sound Design" category, indicating a lack of adequate query expansion or category boosting.

#### User Experience Pain Points
- Creator spam - Multiple nearly identical products by the same creator appearing in results
- Lack of grouping options for similar items
- Overlapping products between "Best-selling" and "Hot and new" sections
- Inconsistent ranking logic for free vs. paid items

## What Has Been Done

### Prototype Search Implementation
The prototype search implementation (v0.9+) includes:
- Exact match for queries surrounded by quotes
- A two-phase search process:
  - Phase 1: Combined-field, match phrase, and fuzzy matching to collect samples
  - Phase 2: Cosine batch with averaged uncompressed ColBERT embeddings with rating boosting
- CLIP text-image embeddings as a fallback model

For similar items recommendation, the prototype uses an empirically weighted combination of:
- CLIP Embedding + Ratings
- Fuzzy match text search

### Current Model Architecture
The prototype system is containerized with the following services:
- Ingestion service
- Core machine learning (ML) service
- API service
- Elasticsearch (v8.17)
- Frontend service (Node & ReactJS)
- NGINX

Caching is implemented at multiple levels: front-end, NGINX, and API services.

### Evaluation Metrics
Evaluation of the prototype compared to the existing search showed significant improvements:
![search_metrics_summary_table](https://github.com/user-attachments/assets/18237acb-31f4-402e-908f-3c2ffbc8612d)

## Proposed Improvements

### Quick Wins
1. **Field Weight Adjustment**: Boost the product title field weight in search ranking. Align field importance as: Product Title > Offer section > Product Description > Requirements > Creator Info > Ratings.

2. **Fuzzy Matching Implementation**: Implement Levenshtein or Damerau-Levenshtein distance for handling misspellings and typos.

3. **Language Detection**: Implement language identification and multi-language decompounding for improved international user experience.

4. **Query Expansion**: Add synonym features to prioritize related terms, especially for category and vertical matching.

5. **Result Grouping**: Group products by creator/seller to avoid search result spam, particularly for identical products from the same creator.

6. **Exact Match Support**: Allow users to type in quotes for exact matching.

### Mid-term Improvements

1. **Hybrid Search Model**: Implement a two-stage ranking system:
   - Stage 1: Efficient model for recall (BM25F)
   - Stage 2: Reranking of top 100-1000 documents with a more complex model

2. **Content-Based Similar Items**: Build a recommendation system using:
   - Text similarity with cached responses
   - Image similarity where appropriate
   - Knowledge graphs or hierarchical structures for better categorization

3. **Time-Aware Recommendations**: Implement an exponential moving average solution that decays users' interest based on their last observation, prioritizing recent activity.

4. **Smart Content Display**: Identify the best product cover image when multiple are available and implement smart cropping for optimal display.

### Advanced Solutions

1. **Neural Collaborative Filtering**: Implement a neural network approach for user-item interaction modeling, particularly for registered users with history.

2. **Graph Neural Networks**: Use GNN architecture to model complex relationships between users, products, and other entities, enabling inductive learning for new items.

3. **Reinforcement Learning**: Implement Contextual Multi-Armed Bandit algorithms for balancing exploration (showing new content) with exploitation (showing proven performers).

4. **RAG-Enhanced Search**: Leverage Retrieval-Augmented Generation concepts:
   - Generate better candidate queries
   - Create internal product descriptions & categories
   - Enhance evaluation tasks

5. **MultiModel & Visual Embeddings**: Implement CLIP or ColPali for unified text-image understanding.

## Technical Implementation Details

### Search and Retrieval Techniques
1. **BM25F Implementation**: Extend Elasticsearch's BM25 to BM25F using combined_fields for better multi-field matching.

2. **Semantic Search**: Use averaged uncompressed ColBERT embeddings or other transformer-based models for semantic understanding.

3. **Learning to Rank**: Train pairwise or listwise LTR models to optimize result ranking based on user interactions.

4. **Visual Search**: Implement image similarity using CLIP or SigLIP embeddings for products with minimal text descriptions.

### Recommendation Systems
1. **Product Metadata Utilization**: Extract and leverage all available product metadata:
   - Categories/taxonomies
   - Creator information
   - Ratings and user interaction data
   - Visual similarities

2. **Collaborative Filtering**: Implement Matrix Factorization or Implicit Alternating Least Squares models based on user-item interactions:
   - Purchase: 1.0 point
   - Cart addition: 0.8 point
   - View: 0.5 point
   - Rating: Weighted score

3. **Hybrid Approaches**: Combine content-based and collaborative filtering methods with side information for comprehensive recommendations.

### Personalization Approaches
1. **Session-Based Personalization**: Track and analyze current session activity to provide contextually relevant recommendations.

2. **User Embeddings**: Create vector representations of users based on their interaction history.

3. **Contextual Multi-Armed Bandit**: Balance exploration and exploitation in recommendations with a focus on maximizing long-term engagement.

## Supporting Notes

### Data Engineering Considerations
1. **Available Data Sources**:
   - Purchase records
   - Search queries
   - User session data
   - Product metadata
   - Creator information

2. **Tracking Requirements**:
   - Query text
   - Session information
   - Product views
   - Selection events
   - Purchase events

3. **Data Generation Strategies**:
   - Summary-based query generation
   - Synthetic evaluation data
   - LLM-based ranking evaluation

### Deployment Considerations
1. **Architecture Recommendations**:
   - Core application layer for basic requests
   - ML service layer for complex operations
   - Proper CI/CD pipeline for model deployment

2. **Performance Optimization**:
   - Caching at multiple levels
   - Proper index sharding
   - Efficient vector retrieval

### Experimental Design
1. **A/B Testing Framework**:
   - Test multiple models simultaneously
   - Session/user/page level testing
   - Multi-Arm Bandit approach for optimization

2. **Evaluation Metrics**:
   - NDCG@k, MAP@k for ranking quality
   - Recall@k, Precision@k for retrieval effectiveness
   - Business metrics: CTR, conversion rate, revenue per session

3. **User Segmentation**:
   - Guest users vs. registered users
   - Creators vs. consumers
   - Geographic and language-based segments
