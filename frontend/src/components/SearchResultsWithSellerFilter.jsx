import React, { useState, useEffect, useCallback } from 'react';
import { ChevronDown, ChevronUp, Filter, Users, Grid } from 'lucide-react';

// Utility functions moved outside of components for reuse
const getEnhancedScore = (products) => {
  const validScores = products.filter(p => p.score != null);
  
  if (validScores.length === 0) return null;
  
  // Calculate base average score
  const sum = validScores.reduce((acc, product) => acc + product.score, 0);
  const avgScore = sum / validScores.length;
  
  // Apply a scaling factor based on the number of products
  // This rewards sellers with multiple relevant products
  // The logarithm ensures the benefit diminishes with scale
  const productCountBonus = Math.log10(1 + validScores.length) * 0.1;
  
  // Combine score and bonus (capped at a reasonable maximum)
  const enhancedScore = Math.min(avgScore * (1 + productCountBonus), 1.0);
  
  return enhancedScore.toFixed(2);
};


const generatePlaceholder = (dim1, dim2, title) => {
  const bgColors = ['212121', '4a4a4a', '6b6b6b', '444', '333', '555', 'abd123', 'fe90ea', '256789', '742d1e'];
  const textColors = ['ffffff', 'f0f0f0', 'eeeeee', 'dddddd', 'cccccc'];

  const bgColor = bgColors[Math.floor(Math.random() * bgColors.length)];
  const textColor = textColors[Math.floor(Math.random() * textColors.length)];


  return `https://placehold.co/${dim1}x${dim2}/${bgColor}/${textColor}?text=${title}`
}

const getAverageRating = (products) => {
  const validRatings = products.filter(p => p.ratings_score != null && p.ratings_score > 0);
  if (validRatings.length === 0) return null;
  const sum = validRatings.reduce((acc, product) => acc + product.ratings_score, 0);
  return (sum / validRatings.length).toFixed(1);
};

const getAverageScore = (products) => {
  const validScores = products.filter(p => p.score != null);
  if (validScores.length === 0) return null;
  const sum = validScores.reduce((acc, product) => acc + product.score, 0);
  return (sum / validScores.length).toFixed(2);
};

const getValidImage = (product) => {
  return product.thumbnail_url || null;
};

const prioritizeValidImages = (products) => {
  const sortedProducts = [...products];
  sortedProducts.sort((a, b) => {
    const aHasImage = !!getValidImage(a);
    const bHasImage = !!getValidImage(b);
    if (aHasImage && !bHasImage) return -1;
    if (!aHasImage && bHasImage) return 1;
    return (b.score || 0) - (a.score || 0);
  });
  return sortedProducts;
};

// Single Seller Card component for displaying one seller
const SellerCard = ({ 
  seller, 
  darkMode, 
  handleSellerClick,
  renderProductCard,
  onHover,
  onLeave
}) => {
  const products = seller.products || [];
  const avgRating = getAverageRating(products);
  const avgScore = getAverageScore(products);
  const prioritizedProducts = prioritizeValidImages(products);
  const bestProduct = prioritizedProducts[0] || products[0];
  
  // Local state for hover effects within the card
  const [hoveredProduct, setHoveredProduct] = useState(null);
  
  const handleLocalProductHover = (productId) => {
    setHoveredProduct(productId);
  };
  
  const handleLocalProductMouseLeave = () => {
    setHoveredProduct(null);
  };
  
  return (
    <div 
      className={`${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-50 hover:bg-gray-100'} 
        rounded-lg overflow-hidden cursor-pointer shadow-sm transition-all hover:shadow-md relative product-card`}
      onClick={() => handleSellerClick(seller.id)}
      onMouseEnter={(e) => onHover && onHover(seller, e, true)}
      onMouseLeave={onLeave}
    >
      {/* Score badge - for consistency with product cards */}
      <div className="absolute top-2 left-2 bg-white/90 dark:bg-gray-800/90 py-0.5 px-1.5 rounded text-xs font-medium flex items-center z-40">
        <span>Score: </span>
        <span className="text-[#FE90EA] ml-1">
          {seller.compositeScore ? seller.compositeScore.toFixed(2) : "N/A"}
        </span>
      </div>
      
      {/* Product image grid - different layout for mobile vs desktop */}
      <div className={`relative group`}>
        {/* Mobile layout (up to md) - adaptive grid based on product count */}
        <div className="md:hidden p-1">
          {/* If 2 products, show them based on their aspect ratio */}
          {products.length === 2 && (
            <div className="grid grid-cols-1 gap-1">
              {prioritizedProducts.map((product, idx) => (
                <div 
                  key={`sm-${product.id}-${idx}`} 
                  className="aspect-video overflow-hidden relative"
                >
                  <img 
                    src={product.thumbnail_url || generatePlaceholder(300,150, product.name)}
                    alt={product.name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      e.target.src = generatePlaceholder(300,150, product.name);
                    }}
                  />
                  {/* Price tag */}
                  {product.price_cents !== undefined && (
                    <div className="absolute rounded-md top-2 right-2 flex items-center" style={{ zIndex: 30 }}>
                      <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                        ${(product.price_cents / 100).toFixed(2)}
                        <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                        <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          
          {/* If 3 products, show one large on top, two smaller below */}
          {products.length === 3 && (
            <div className="flex flex-col gap-1">
              <div className="aspect-video overflow-hidden relative">
                <img 
                  src={prioritizedProducts[0].thumbnail_url || generatePlaceholder(400,200, prioritizedProducts[0].name)}
                  alt={prioritizedProducts[0].name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = generatePlaceholder(400,200, prioritizedProducts[0].name);
                  }}
                />
                {/* Price tag */}
                {prioritizedProducts[0].price_cents !== undefined && (
                  <div className="absolute rounded-md top-2 right-2 flex items-center" style={{ zIndex: 30 }}>
                    <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                      ${(prioritizedProducts[0].price_cents / 100).toFixed(2)}
                      <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                      <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                    </div>
                  </div>
                )}
              </div>
              <div className="grid grid-cols-2 gap-1">
                {prioritizedProducts.slice(1, 3).map((product, idx) => (
                  <div 
                    key={`sm-${product.id}-${idx}`} 
                    className="aspect-video overflow-hidden relative"
                  >
                    <img 
                      src={product.thumbnail_url || generatePlaceholder(200,100, product.name)}
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = generatePlaceholder(200,100, product.name);
                      }}
                    />
                    {/* Price tag */}
                    {product.price_cents !== undefined && (
                      <div className="absolute rounded-md top-2 right-2 flex items-center" style={{ zIndex: 30 }}>
                        <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                          ${(product.price_cents / 100).toFixed(2)}
                          <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                          <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* If 4 or more products, show featured layout with indicator */}
          {products.length >= 4 && (
            <div className="flex flex-col gap-1">
              <div className="aspect-video overflow-hidden relative">
                <img 
                  src={prioritizedProducts[0].thumbnail_url || generatePlaceholder(300,150, prioritizedProducts[0].name)}
                  alt={prioritizedProducts[0].name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = generatePlaceholder(300,150, prioritizedProducts[0].name);
                  }}
                />
                {/* Price tag */}
                {prioritizedProducts[0].price_cents !== undefined && (
                  <div className="absolute rounded-md top-2 right-2 flex items-center" style={{ zIndex: 30 }}>
                    <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                      ${(prioritizedProducts[0].price_cents / 100).toFixed(2)}
                      <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                      <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                    </div>
                  </div>
                )}
              </div>
              <div className="grid grid-cols-3 gap-1">
                {prioritizedProducts.slice(1, 4).map((product, idx) => (
                  <div 
                    key={`sm-${product.id}-${idx}`} 
                    className="aspect-square overflow-hidden relative"
                  >
                    <img 
                      src={product.thumbnail_url || generatePlaceholder(100,100, product.name)}
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = generatePlaceholder(100,100, product.name);
                      }}
                    />
                    {/* Price tag - only show if not the last one with +N overlay */}
                    {product.price_cents !== undefined && !(idx === 2 && products.length > 4) && (
                      <div className="absolute rounded-md top-1 right-1 flex items-center" style={{ zIndex: 30 }}>
                        <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                          ${(product.price_cents / 100).toFixed(2)}
                          <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                          <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                        </div>
                      </div>
                    )}
                    {idx === 2 && products.length > 4 && (
                      <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                        <span className="text-white text-sm font-bold">+{products.length - 4}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* Desktop layout (md and up) - adaptive grid based on product count */}
        <div className="hidden md:block p-1">
          {/* If 2 products, show one over the other */}
          {products.length === 2 && (
            <div className="flex flex-col gap-1 h-64">
              {prioritizedProducts.map((product, idx) => (
                <div 
                  key={`md-${product.id}-${idx}`}
                  className="flex-1 overflow-hidden relative"
                >
                  <img 
                    src={product.thumbnail_url || generatePlaceholder(400,150, product.name)}
                    alt={product.name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      e.target.src = generatePlaceholder(400,150, product.name);
                    }}
                  />
                  {/* Price tag */}
                  {product.price_cents !== undefined && (
                    <div className="absolute rounded-md top-2 right-2 flex items-center" style={{ zIndex: 30 }}>
                      <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                        ${(product.price_cents / 100).toFixed(2)}
                        <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                        <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          
          {/* For desktop with 3+ products, show one large, two smaller */}
          {products.length >= 3 && (
            <div className="flex flex-col gap-1 h-64">
              <div className="flex-1 overflow-hidden relative">
                <img 
                  src={prioritizedProducts[0].thumbnail_url || generatePlaceholder(400,150, prioritizedProducts[0].name)}
                  alt={prioritizedProducts[0].name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = generatePlaceholder(400,150, prioritizedProducts[0].name);
                  }}
                />
                {/* Price tag */}
                {prioritizedProducts[0].price_cents !== undefined && (
                  <div className="absolute rounded-md top-2 right-2 flex items-center" style={{ zIndex: 30 }}>
                    <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                      ${(prioritizedProducts[0].price_cents / 100).toFixed(2)}
                      <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                      <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                    </div>
                  </div>
                )}
              </div>
              <div className="grid grid-cols-2 gap-1 flex-1">
                {prioritizedProducts.slice(1, 3).map((product, idx) => (
                  <div 
                    key={`md-sm-${product.id}-${idx}`}
                    className="overflow-hidden relative"
                  >
                    <img 
                      src={product.thumbnail_url || generatePlaceholder(200,150, product.name)}
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = generatePlaceholder(200,150, product.name);
                      }}
                    />
                    {/* Price tag */}
                    {product.price_cents !== undefined && !(idx === 1 && products.length > 3) && (
                      <div className="absolute rounded-md top-2 right-2 flex items-center" style={{ zIndex: 30 }}>
                        <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                          ${(product.price_cents / 100).toFixed(2)}
                          <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                          <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                        </div>
                      </div>
                    )}
                    {idx === 1 && products.length > 3 && (
                      <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                        <span className="text-white text-lg font-bold">+{products.length - 3}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* Best product title overlay at the bottom with swipe animation */}
        <div className="absolute bottom-0 left-0 right-0 bg-black p-2 overflow-hidden"
            style={{ opacity: 0.8 }}
          >
            <div className="relative h-5 overflow-hidden">
              {prioritizedProducts.slice(0, 4).map((product, idx) => (
                <h3 
                  key={`title-${product.id || idx}`} 
                  className="absolute inset-x-0 text-white text-xs font-medium truncate whitespace-nowrap"
                  style={{
                    animation: `titleSwipe 12s linear infinite ${idx * 3}s`,
                    opacity: idx === 0 ? 1 : 0,
                    transform: idx === 0 ? 'translateX(0)' : 'translateX(100%)'
                  }}
                >
                  {product.name}
                </h3>
              ))}
            </div>
          </div>

          {/* Add animation keyframes */}
          <style dangerouslySetInnerHTML={{
            __html: `
              @keyframes titleSwipe {
                0% { transform: translateX(100%); opacity: 0; }
                5%, 20% { transform: translateX(0); opacity: 1; }
                25%, 100% { transform: translateX(-100%); opacity: 0; }
              }
            `
          }} />
        
        {/* Hover effect for product titles */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
          <div 
            className="absolute bottom-0 left-0 right-0 p-3 backdrop-blur-sm bg-black/50"
            style={{
              opacity: 0.9,
            }}
          >
            <h3 className="font-medium text-sm text-gray-100 mb-1 line-clamp-1">{bestProduct.name}</h3>
          </div>
        </div>
      </div>
      
      {/* Seller info at the bottom - compact design */}
      <div className="p-2 border-t border-gray-600 flex items-center justify-between">
        <div className="flex items-center flex-grow overflow-hidden">
          {seller.thumbnail && (
            <div className="w-5 h-5 rounded-full overflow-hidden mr-1 flex-shrink-0">
              <img 
                src={seller.thumbnail} 
                alt={seller.name}
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.target.src = `https://placehold.co/100x100?text=${seller.name.charAt(0)}`;
                }}
              />
            </div>
          )}
          
          <span className={`text-xs truncate ${darkMode ? 'text-gray-300' : 'text-gray-700'} mr-1 max-w-[80px]`}>
            {seller.name}
          </span>
          
          {/* Compact rating with count */}
          {avgRating && (
            <div className="flex items-center text-yellow-400 ml-auto mr-1">
              <span>★</span>
              <span className={`text-xs ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                {avgRating}
              </span>
              <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'} ml-1`}>
                ({products.filter(p => p.ratings_score != null && p.ratings_score > 0).length})
              </span>
            </div>
          )}
          
          {/* Compact score badge */}
          {avgScore && (
            <div className="flex items-center">
              <div className="px-1 py-0.5 bg-[#FE90EA] text-black rounded-full text-xs ml-1 flex-shrink-0">
                {seller.compositeScore ? seller.compositeScore.toFixed(2) : getEnhancedScore(products)}
              </div>
              {products.length > 1 && seller.simpleAvgScore && (
                <div className="ml-1 text-xs text-gray-400" title="Boost from multiple products">
                  +{Math.round((
                    (seller.compositeScore || parseFloat(getEnhancedScore(products))) - 
                    seller.simpleAvgScore
                  ) * 100)}%
                </div>
              )}
            </div>
          )}
        </div>
        
        <div className="flex items-center">
          {/* Blue tag showing n+ for products count when there are at least 2 products */}
          {products.length >= 2 ? (
            <div className="px-1.5 py-0.5 bg-blue-500 text-white rounded-full text-xs flex-shrink-0">
              {products.length-1}+ more
            </div>
          ) : (
            <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {products.length} product
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

// Multiple Seller Cards component (for backward compatibility)
const SellerCards = ({ 
  sellers, 
  darkMode, 
  handleSellerClick,
  searchResults,
  renderProductCard,
  onHover,
  onLeave
}) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
      {sellers.map(seller => {
        // If there's only one product, render the ProductCard directly
        if (seller.products?.length === 1 && renderProductCard) {
          const product = seller.products[0];
          product.displayScore = seller.compositeScore || product.score;
          return renderProductCard(product, 0);
        }
        
        // Otherwise render a seller card
        return (
          <SellerCard
            key={seller.id}
            seller={seller}
            darkMode={darkMode}
            handleSellerClick={handleSellerClick}
            renderProductCard={renderProductCard}
            onHover={onHover}
            onLeave={onLeave}
          />
        );
      })}
    </div>
  );
};

// Main Search Results component with unified blending
const SearchResultsWithSellerFilter = ({ 
  searchResults, 
  darkMode, 
  isLoading,
  query,
  renderProductCard,
  onHover,
  onLeave
}) => {
  const [groupBySeller, setGroupBySeller] = useState(true);
  const [selectedSeller, setSelectedSeller] = useState(null);
  const [sellerGroups, setSellerGroups] = useState([]);
  const [unifiedItems, setUnifiedItems] = useState([]);
  
  // Process search results to create seller groups and unified items
  useEffect(() => {
    if (searchResults && searchResults.length > 0) {
      // Create seller groups from search results
      const groups = {};
      
      searchResults.forEach(product => {
        // Extract seller info from product
        const sellerId = product.seller_id || 
                        (product.seller_name ? `seller-${product.seller_name}` : 'unknown');
        const sellerName = product.seller_name || 'Unknown Seller';
        
        if (!groups[sellerId]) {
          groups[sellerId] = {
            id: sellerId,
            name: sellerName,
            thumbnail: product.seller_thumbnail || null,
            products: [],
            avgScore: 0
          };
        }
        
        groups[sellerId].products.push(product);
      });
      
      // Calculate scores for each seller using enhanced methods
      Object.values(groups).forEach(seller => {
        const validScores = seller.products.filter(p => p.score != null);
        if (validScores.length > 0) {
          // Store simple average for reference
          seller.simpleAvgScore = validScores.reduce((acc, product) => 
            acc + product.score, 0) / validScores.length;
          
          // Enhanced score with product count bonus
          const productCountBonus = Math.log10(1 + validScores.length) * 0.1;
          seller.enhancedScore = seller.simpleAvgScore * (1 + productCountBonus);
          
          // Max product score for tie-breaking
          seller.maxProductScore = Math.max(...validScores.map(p => p.score || 0));
          
          // Create composite score for sorting
          seller.compositeScore = (
            seller.enhancedScore * 0.7 +   // 70% weight to enhanced score
            seller.maxProductScore * 0.3    // 30% weight to best product
          );
        } else {
          seller.simpleAvgScore = 0;
          seller.enhancedScore = 0;
          seller.maxProductScore = 0;
          seller.compositeScore = 0;
        }
      });
      
      // Set the seller groups (for original grouping functionality)
      setSellerGroups(Object.values(groups));
      
      // Create unified items list for blended display
      const items = [];
      
      Object.values(groups).forEach(seller => {
        if (seller.products.length === 1) {
          // For single product sellers, add the product directly
          const product = seller.products[0];
          product.sellerCompositeScore = seller.compositeScore;
          product.displayScore = seller.compositeScore || product.score;
          items.push({ 
            type: 'product', 
            data: product,
            sortScore: product.displayScore
          });
        } else {
          // For multi-product sellers, add the seller group
          items.push({ 
            type: 'seller', 
            data: seller,
            sortScore: seller.compositeScore
          });
        }
      });
      
      // Sort all items together by their sort score
      items.sort((a, b) => b.sortScore - a.sortScore);
      
      // Debug log to check sorting
      console.log('Unified sorted items:', items.slice(0, 5).map(item => ({
        type: item.type,
        name: item.type === 'product' ? item.data.name : item.data.name,
        products: item.type === 'seller' ? item.data.products.length : 1,
        score: item.sortScore?.toFixed(2)
      })));
      
      // Set the unified items
      setUnifiedItems(items);
      
    } else {
      setSellerGroups([]);
      setUnifiedItems([]);
    }
  }, [searchResults]);
  
  // Handle seller card click
  const handleSellerClick = useCallback((sellerId) => {
    if (selectedSeller === sellerId) {
      setSelectedSeller(null);
    } else {
      setSelectedSeller(sellerId);
    }
  }, [selectedSeller]);
  
  // Get filtered results based on selected seller
  const getFilteredResults = useCallback(() => {
    if (!selectedSeller) return searchResults;
    
    return searchResults.filter(product => 
      product.seller_id === selectedSeller || 
      (product.seller_name && `seller-${product.seller_name}` === selectedSeller)
    );
  }, [searchResults, selectedSeller]);
  
  // Toggle group by seller view
  const toggleGroupBySeller = () => {
    setGroupBySeller(!groupBySeller);
    setSelectedSeller(null); // Clear any selected seller when toggling
  };
  
  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#FE90EA]"></div>
      </div>
    );
  }
  
  // If no search results, show empty state
  if (!searchResults || searchResults.length === 0) {
    return (
      <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-6 rounded-lg shadow-sm text-center`}>
        <p className={`text-lg ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          No results found for "{query}". Try a different search term.
        </p>
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      {/* Results header with toggle for grouping by seller */}
      <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 rounded-lg shadow-sm`}>
        <div className="flex flex-wrap justify-between items-center">
          <h2 className={`text-lg sm:text-xl font-semibold ${darkMode ? 'text-white' : 'text-black'} border-b-2 border-[#FE90EA] pb-2 inline-block`}>
            {selectedSeller 
              ? `${sellerGroups.find(s => s.id === selectedSeller)?.name}'s Products` 
              : `Search Results (${searchResults.length})`
            }
          </h2>
          
          <div className="flex items-center space-x-4 mt-2 sm:mt-0">
            {selectedSeller && (
              <button
                onClick={() => setSelectedSeller(null)}
                className="text-sm text-[#FE90EA] hover:underline"
              >
                View all results
              </button>
            )}
            
            {/* Toggle switch for grouping by seller */}
            <div className="flex items-center">
              <span className={`text-sm mr-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Group by Seller</span>
              
              <button 
                onClick={toggleGroupBySeller}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                  groupBySeller 
                    ? 'bg-[#FE90EA]' 
                    : darkMode ? 'bg-gray-600' : 'bg-gray-300'
                }`}
                role="switch"
                aria-checked={groupBySeller}
              >
                <span 
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                    groupBySeller ? 'translate-x-6' : 'translate-x-1'
                  }`} 
                />
              </button>
              
              {/* Toggle icon */}
              <span className="ml-2">
                {groupBySeller ? (
                  <Users size={16} className={darkMode ? 'text-white' : 'text-black'} />
                ) : (
                  <Grid size={16} className={darkMode ? 'text-white' : 'text-black'} />
                )}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Results grid - now displays either grouped or individual products */}
      <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 rounded-lg shadow-sm`}>
        {/* When a seller is selected, show just their products */}
        {selectedSeller ? (
          <div className="grid gap-3 sm:gap-6 grid-cols-2 xl:grid-cols-3 product-grid">
            {getFilteredResults().map((product, index) => renderProductCard(product, index))}
          </div>
        ) : (
          /* Otherwise show either grouped sellers or individual products based on toggle */
          groupBySeller ? (
            /* Grouped view - show seller cards */
            <SellerCards 
              sellers={sellerGroups} 
              darkMode={darkMode} 
              handleSellerClick={handleSellerClick}
              searchResults={searchResults}
              renderProductCard={renderProductCard}
              onHover={onHover}
              onLeave={onLeave}
            />
          ) : (
            /* Individual products view - just show all products sorted by score */
            <div className="grid gap-3 sm:gap-6 grid-cols-2 xl:grid-cols-3 product-grid">
              {searchResults
                .slice() // Create a copy to avoid mutating the original
                .sort((a, b) => (b.score || 0) - (a.score || 0)) // Sort by score
                .map((product, index) => renderProductCard(product, index))
              }
            </div>
          )
        )}
      </div>
    </div>
  );
};

export { SearchResultsWithSellerFilter, SellerCard, SellerCards };
export default SearchResultsWithSellerFilter;