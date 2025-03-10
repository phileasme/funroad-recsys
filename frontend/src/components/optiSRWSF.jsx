const normalizedResults = useMemo(() => {
  // Start with individual products
  const productResults = (selectedSeller ? filteredResults : searchResults)
    .filter(product => {
      const score = product.score ? parseFloat(product.score) : 0;
      return score >= 0.55;
    })
    .map(product => ({
      type: 'product',
      data: product,
      score: parseFloat(product.score || 0),
      rating: product.bayesianRating || 0,
      ratingCount: product.ratings_count || 0,
      price: product.price_cents !== undefined ? product.price_cents : Number.MAX_SAFE_INTEGER
    }));

  // Prepare seller groups
  const sellerGroupResults = sellerGroups
    .filter(seller => {
      // Check if seller has any products above threshold
      const validProducts = seller.products.filter(product => {
        const score = product.score ? parseFloat(product.score) : 0;
        return score >= 0.55;
      });
      return validProducts.length > 0;
    })
    .map(seller => ({
      type: 'seller',
      data: seller,
      score: seller.combinedScore || 0,
      rating: seller.avgRating || 0,
      ratingCount: seller.totalRatingsCount || 0,
      price: seller.avgPrice !== undefined ? seller.avgPrice : Number.MAX_SAFE_INTEGER,
      productCount: seller.products.filter(product => 
        parseFloat(product.score || 0) >= 0.55
      ).length
    }));

  // Combine both types of results
  return [...productResults, ...sellerGroupResults];
}, [searchResults, filteredResults, sellerGroups, selectedSeller]);

// Sort results based on selected sort order
const sortedResults = useMemo(() => {
  return [...normalizedResults].sort((a, b) => {
    switch (sortOrder) {
      case 'price-asc':
        if (a.price !== b.price) return a.price - b.price;
        return a.score - b.score;
      
      case 'price-desc':
        if (a.price !== b.price) return b.price - a.price;
        return b.score - a.score;
      
      case 'rating':
        // Primary sort by rating
        if (Math.abs(b.rating - a.rating) > 0.01) {
          return b.rating - a.rating;
        }
        
        // Tie-breaker: rating count
        if (a.ratingCount !== b.ratingCount) {
          return b.ratingCount - a.ratingCount;
        }
        
        // Final tie-breaker: score
        return b.score - a.score;
      
      case 'score':
      default:
        // Primary sort by score
        if (Math.abs(b.score - a.score) > 0.0001) {
          return b.score - a.score;
        }
        
        // Tie-breaker: rating
        if (Math.abs(b.rating - a.rating) > 0.01) {
          return b.rating - a.rating;
        }
        
        // Final tie-breaker: rating count
        return b.ratingCount - a.ratingCount;
    }
  });
}, [normalizedResults, sortOrder]);

const paginatedResults = useMemo(() => {
  return sortedResults.slice(0, visibleItems);
}, [sortedResults, visibleItems]);

const renderContent = useMemo(() => {
  return paginatedResults.map((item, index) => {
    if (item.type === 'product') {
      return renderProductCard(item.data, index);
    } else if (item.type === 'seller') {
      return (
        <SellerCards
          sellers={[item.data]}
          darkMode={darkMode}
          handleSellerClick={handleSellerClick}
          renderProductCard={renderProductCard}
          onHover={onHover}
          onLeave={onLeave}
        />
      );
    }
    return null;
  });
}, [paginatedResults, renderProductCard, darkMode, handleSellerClick, onHover, onLeave]);