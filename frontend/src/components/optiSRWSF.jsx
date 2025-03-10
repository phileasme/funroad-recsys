import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { ChevronDown, ChevronUp, Filter, Users, Grid } from 'lucide-react';
import { FixedSizeGrid as Grid } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import memoize from 'memoize-one';


// Utility functions for reuse
const generatePlaceholder = (dim1, dim2, title) => {
    const bgColors = ['212121', '4a4a4a', '6b6b6b', '444', '333', '555', 'abd123', 'fe90ea', '256789', '742d1e'];
    const textColors = ['ffffff', 'f0f0f0', 'eeeeee', 'dddddd', 'cccccc'];
  
    const bgColor = bgColors[Math.floor(Math.random() * bgColors.length)];
    const textColor = textColors[Math.floor(Math.random() * textColors.length)];
  
    return `https://placehold.co/${dim1}x${dim2}/${bgColor}/${textColor}?text=${encodeURIComponent(title || '')}`;
  };
  
  const getAverageRating = (products) => {
    const validRatings = products.filter(p => p.ratings_score != null && p.ratings_score > 0);
    if (validRatings.length === 0) return null;
    const sum = validRatings.reduce((acc, product) => acc + product.ratings_score, 0);
    return (sum / validRatings.length).toFixed(1);
  };
  
  const getAverageScore = (products) => {
    const validScores = products.filter(p => p.score != null);
    if (validScores.length === 0) return null;
    const sum = validScores.reduce((acc, product) => acc + parseFloat(product.score || 0), 0);
    return (sum / validScores.length).toFixed(2);
  };
  
  const prioritizeValidImages = (products) => {
    const sortedProducts = [...products];
    sortedProducts.sort((a, b) => {
      const aHasImage = !!a.thumbnail_url;
      const bHasImage = !!b.thumbnail_url;
      if (aHasImage && !bHasImage) return -1;
      if (!aHasImage && bHasImage) return 1;
      return (parseFloat(b.score || 0) - parseFloat(a.score || 0));
    });
    return sortedProducts;
  };

// Create a virtualized item renderer
const VirtualizedProductGrid = React.memo(({ 
  items, 
  renderItem, 
  darkMode,
  columnCount = 3,
  rowHeight = 350
}) => {
  // Memoize the creation of the items grid for react-window
  const createItemData = memoize((items, renderItem, darkMode) => ({
    items,
    renderItem,
    darkMode
  }));
  
  const itemData = createItemData(items, renderItem, darkMode);
  
  // Inner item renderer for the virtualized list
  const Cell = ({ columnIndex, rowIndex, style, data }) => {
    const { items, renderItem, darkMode } = data;
    const index = rowIndex * columnCount + columnIndex;
    
    // Check if item exists at this index
    if (index >= items.length) {
      return <div style={style} />;
    }
    
    // Apply padding for better visual appearance
    const innerStyle = {
      ...style,
      paddingLeft: columnIndex === 0 ? 0 : 8,
      paddingRight: columnIndex === columnCount - 1 ? 0 : 8,
      paddingTop: rowIndex === 0 ? 0 : 8,
      paddingBottom: 8
    };
    
    return (
      <div style={innerStyle}>
        {renderItem(items[index], index)}
      </div>
    );
  };
  
  // Calculate row count based on item count and column count
  const rowCount = Math.ceil(items.length / columnCount);
  
  return (
    <AutoSizer>
      {({ height, width }) => {
        // Dynamically adjust column count based on width
        let dynamicColumnCount = columnCount;
        if (width < 768) {
          dynamicColumnCount = 2; // 2 columns for small screens
        } else if (width >= 1280) {
          dynamicColumnCount = 3; // 3 columns for large screens
        }
        
        // Recalculate row count based on dynamic column count
        const adjustedRowCount = Math.ceil(items.length / dynamicColumnCount);
        
        return (
          <Grid
            className="virtualized-grid"
            columnCount={dynamicColumnCount}
            columnWidth={width / dynamicColumnCount}
            height={height || 800}
            rowCount={adjustedRowCount}
            rowHeight={rowHeight}
            width={width}
            itemData={itemData}
          >
            {Cell}
          </Grid>
        );
      }}
    </AutoSizer>
  );
});

// Optimized SellerCard component with React.memo
const SellerCard = React.memo(({ seller, darkMode, handleSellerClick, onHover, onLeave }) => {
    const handleClick = useCallback(() => {
      handleSellerClick(seller.id);
    }, [handleSellerClick, seller.id]);
  
    const handleMouseEnter = useCallback((e) => {
      if (onHover) onHover(seller, e, true);
    }, [onHover, seller]);
  return (
    <div
      className={`${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-50 hover:bg-gray-100'} 
        rounded-lg overflow-hidden cursor-pointer shadow-sm transition-all hover:shadow-md relative product-card`}
      onClick={() => handleSellerClick(seller.id)}
      onMouseEnter={(e) => onHover && onHover(seller, e, true)}
      onMouseLeave={onLeave}
    >
            {/* Score badge */}
            <div className="absolute top-2 left-2 bg-white/90 dark:bg-gray-800/90 py-0.5 px-1.5 rounded text-xs font-medium flex items-center z-40">
        <span>Score: </span>
        <span className="text-[#FE90EA] ml-1">
          {seller.compositeScore ? seller.compositeScore.toFixed(2) : "N/A"}
        </span>
      </div>

      {/* Product image grid - adaptive based on screen size and product count */}
      <div className="relative group">
        {/* Mobile layout (up to md) */}
        <div className="md:hidden p-1">
          {products.length === 2 && (
            <div className="grid grid-cols-1 gap-1">
              {prioritizedProducts.map((product, idx) => (
                <div key={`sm-${product.id || idx}-${idx}`} className="aspect-video overflow-hidden relative">
                  <img
                    src={product.thumbnail_url || generatePlaceholder(300, 150, product.name)}
                    alt={product.name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      e.target.src = generatePlaceholder(300, 150, product.name);
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

          {products.length === 3 && (
            <div className="flex flex-col gap-1">
              <div className="aspect-video overflow-hidden relative">
                <img
                  src={prioritizedProducts[0].thumbnail_url || generatePlaceholder(400, 200, prioritizedProducts[0].name)}
                  alt={prioritizedProducts[0].name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = generatePlaceholder(400, 200, prioritizedProducts[0].name);
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
                  <div key={`sm-${product.id || idx}-${idx}`} className="aspect-video overflow-hidden relative">
                    <img
                      src={product.thumbnail_url || generatePlaceholder(200, 100, product.name)}
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = generatePlaceholder(200, 100, product.name);
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

          {products.length >= 4 && (
            <div className="flex flex-col gap-1">
              <div className="aspect-video overflow-hidden relative">
                <img
                  src={prioritizedProducts[0].thumbnail_url || generatePlaceholder(300, 150, prioritizedProducts[0].name)}
                  alt={prioritizedProducts[0].name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = generatePlaceholder(300, 150, prioritizedProducts[0].name);
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
                  <div key={`sm-${product.id || idx}-${idx}`} className="aspect-square overflow-hidden relative">
                    <img
                      src={product.thumbnail_url || generatePlaceholder(100, 100, product.name)}
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = generatePlaceholder(100, 100, product.name);
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

        {/* Desktop layout (md and up) */}
        <div className="hidden md:block p-1">
          {/* Layout for 2+ products */}
          <div className="flex flex-col gap-1 h-64">
            <div className="flex-1 overflow-hidden relative">
              <img
                src={prioritizedProducts[0]?.thumbnail_url || generatePlaceholder(400, 150, prioritizedProducts[0]?.name)}
                alt={prioritizedProducts[0]?.name}
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.target.src = generatePlaceholder(400, 150, prioritizedProducts[0]?.name);
                }}
              />
              {/* Price tag */}
              {prioritizedProducts[0]?.price_cents !== undefined && (
                <div className="absolute rounded-md top-2 right-2 flex items-center" style={{ zIndex: 30 }}>
                  <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                    ${(prioritizedProducts[0].price_cents / 100).toFixed(2)}
                    <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                    <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                  </div>
                </div>
              )}
            </div>
            
            {products.length > 1 && (
              <div className="grid grid-cols-2 gap-1 flex-1">
                {prioritizedProducts.slice(1, 3).map((product, idx) => (
                  <div key={`md-sm-${product.id || idx}-${idx}`} className="overflow-hidden relative">
                    <img
                      src={product.thumbnail_url || generatePlaceholder(200, 150, product.name)}
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = generatePlaceholder(200, 150, product.name);
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
            )}
          </div>
        </div>

        {/* Best product title overlay with swipe animation */}
        <div className="absolute bottom-0 left-0 right-0 bg-black p-2 overflow-hidden" style={{ opacity: 0.8 }}>
          <div className="relative h-5 overflow-hidden">
            {prioritizedProducts.slice(0, 4).map((product, idx) => (
              <h3
                key={`title-${product.id || idx}`}
                className="absolute inset-x-0 text-white text-xs font-medium truncate whitespace-nowrap"
                style={{
                  animation: `titleSwipe 12s linear infinite ${idx * 3}s`,
                  opacity: idx === 0 ? 1 : 0,
                  transform: idx === 0 ? "translateX(0)" : "translateX(100%)",
                }}
              >
                {product.name}
              </h3>
            ))}
          </div>
        </div>

        {/* Add animation keyframes */}
        <style
          dangerouslySetInnerHTML={{
            __html: `
              @keyframes titleSwipe {
                0% { transform: translateX(100%); opacity: 0; }
                5%, 20% { transform: translateX(0); opacity: 1; }
                25%, 100% { transform: translateX(-100%); opacity: 0; }
              }
            `,
          }}
        />

        {/* Hover effect */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="absolute bottom-0 left-0 right-0 p-3 backdrop-blur-sm bg-black/50" style={{ opacity: 0.9 }}>
            <h3 className="font-medium text-sm text-gray-100 mb-1 line-clamp-1">{bestProduct?.name}</h3>
          </div>
        </div>
      </div>

      {/* Seller info footer */}
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
        </div>

        <div className="flex items-center">
          {/* Product count badge */}
          {products.length >= 2 ? (
            <div className="px-1.5 py-0.5 bg-blue-500 text-white rounded-full text-xs flex-shrink-0">
              {products.length - 1}+ more
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
});

// Main component with virtualization
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
  const [isMobile, setIsMobile] = useState(false);
  
  // Check for mobile viewport
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };
    
    handleResize(); // Set initial value
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Process search results to create seller groups - keep your existing logic
  useEffect(() => {
    if (!searchResults || searchResults.length === 0) {
      setSellerGroups([]);
      return;
    }
    
 
    // Create seller groups from search results
    const groups = {};
    
    searchResults.forEach((product) => {
      // Extract seller info from product
      const sellerId = product.seller_id || 
                      (product.seller_name ? `seller-${product.seller_name}` : "unknown");
      const sellerName = product.seller_name || "Unknown Seller";
      
      if (!groups[sellerId]) {
        groups[sellerId] = {
          id: sellerId,
          name: sellerName,
          thumbnail: product.seller_thumbnail || null,
          products: [],
          avgScore: 0,
        };
      }
      
      groups[sellerId].products.push(product);
    });
    
    // Calculate scores for each seller
    Object.values(groups).forEach((seller) => {
      const validScores = seller.products.filter((p) => p.score != null);
      if (validScores.length > 0) {
        // Store simple average for reference
        seller.simpleAvgScore = validScores.reduce((acc, product) => 
          acc + parseFloat(product.score || 0), 0) / validScores.length;
        
        // Enhanced score with product count bonus
        const productCountBonus = Math.log10(1 + validScores.length) * 0.1;
        seller.enhancedScore = seller.simpleAvgScore * (1 + productCountBonus);
        
        // Max product score for tie-breaking
        seller.maxProductScore = Math.max(...validScores.map((p) => parseFloat(p.score || 0)));
        
        // Composite score for sorting
        seller.compositeScore = seller.enhancedScore * 0.7 + seller.maxProductScore * 0.3;
      } else {
        seller.simpleAvgScore = 0;
        seller.enhancedScore = 0;
        seller.maxProductScore = 0;
        seller.compositeScore = 0;
      }
    });
    
    // Set seller groups
    setSellerGroups(Object.values(groups));
  }, [searchResults]);
  
  // Memoize handlers to prevent recreations
  const handleSellerClick = useCallback((sellerId) => {
    setSelectedSeller(prev => prev === sellerId ? null : sellerId);
  }, []);
  
  const toggleGroupBySeller = useCallback(() => {
    setGroupBySeller(prev => !prev);
    setSelectedSeller(null);
  }, []);
  
  // Memoize filtered results
  const filteredResults = useMemo(() => {
    if (!selectedSeller) return searchResults;
    
    return searchResults.filter(product => 
      product.seller_id === selectedSeller || 
      (product.seller_name && `seller-${product.seller_name}` === selectedSeller)
    );
  }, [searchResults, selectedSeller]);
  
  // Handle loading and empty states
  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#FE90EA]"></div>
      </div>
    );
  }
  
  if (!searchResults || searchResults.length === 0) {
    return (
      <div className={`${darkMode ? "bg-gray-800" : "bg-white"} p-6 rounded-lg shadow-sm text-center`}>
        <p className={`text-lg ${darkMode ? "text-gray-300" : "text-gray-600"}`}>
          No results found for "{query}". Try a different search term.
        </p>
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      {/* Results header with toggle for grouping by seller */}
          <div className={`${darkMode ? "bg-gray-800" : "bg-white"} p-4 rounded-lg shadow-sm`}>
              <div className="flex flex-wrap justify-between items-center">
                <h2 className={`text-lg sm:text-xl font-semibold ${darkMode ? "text-white" : "text-black"} border-b-2 border-[#FE90EA] pb-2 inline-block`}>
                  {selectedSeller
                    ? `${sellerGroups.find((s) => s.id === selectedSeller)?.name}'s Products`
                    : `Search Results (${searchResults.length})`}
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
                    <span className={`text-sm mr-2 ${darkMode ? "text-gray-300" : "text-gray-600"}`}>
                      Group by Seller
                    </span>
      
                    <button
                      onClick={toggleGroupBySeller}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                        groupBySeller
                          ? "bg-[#FE90EA]"
                          : darkMode
                          ? "bg-gray-600"
                          : "bg-gray-300"
                      }`}
                      role="switch"
                      aria-checked={groupBySeller}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                          groupBySeller ? "translate-x-6" : "translate-x-1"
                        }`}
                      />
                    </button>
      
                    {/* Toggle icon */}
                    <span className="ml-2">
                      {groupBySeller ? (
                        <Users size={16} className={darkMode ? "text-white" : "text-black"} />
                      ) : (
                        <Grid size={16} className={darkMode ? "text-white" : "text-black"} />
                      )}
                    </span>
                  </div>
                </div>
              </div>
            </div>

      {/* Results grid with virtualization */}
      <div 
        className={`${darkMode ? "bg-gray-800" : "bg-white"} p-4 rounded-lg shadow-sm`}
        style={{ height: isMobile ? '70vh' : '80vh' }} // Fixed height for virtualization
      >
        {selectedSeller ? (
          <VirtualizedProductGrid
            items={filteredResults}
            renderItem={renderProductCard}
            darkMode={darkMode}
            columnCount={isMobile ? 2 : 3}
          />
        ) : groupBySeller ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {sellerGroups.map((seller) => (
              <SellerCard
                key={seller.id}
                seller={seller}
                darkMode={darkMode}
                handleSellerClick={handleSellerClick}
                onHover={onHover}
                onLeave={onLeave}
              />
            ))}
          </div>
        ) : (
          <VirtualizedProductGrid
            items={searchResults}
            renderItem={renderProductCard}
            darkMode={darkMode}
            columnCount={isMobile ? 2 : 3}
          />
        )}
      </div>
    </div>
  );
};

export default React.memo(SearchResultsWithSellerFilter);