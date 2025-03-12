import React, { useState, useEffect, useMemo, useCallback, useRef, Suspense } from 'react';
import { ChevronDown, ChevronUp, Filter, Users, Grid, List, Search } from 'lucide-react';
import { useInView } from 'react-intersection-observer';
import { FixedSizeGrid as FGrid } from 'react-window';
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


const prioritizeValidImages = (products) => {
  if (!Array.isArray(products) || products.length === 0) return [];
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

const OptimizedLoadMore = React.memo(({ onLoadMore, hasMore, isLoading, darkMode }) => {
  // Use intersection observer with specific thresholds
  const { ref, inView } = useInView({
    threshold: 0.1,
    rootMargin: '200px 0px',
    triggerOnce: false
  });

  // Track the last time we loaded more to prevent rapid firing
  const lastLoadRef = useRef(Date.now());
  
  // Effect to trigger load more when element comes into view
  useEffect(() => {
    if (inView && hasMore && !isLoading) {
      const now = Date.now();
      // Prevent loading more than once every 500ms
      if (now - lastLoadRef.current > 500) {
        lastLoadRef.current = now;
        onLoadMore();
      }
    }
  }, [inView, hasMore, isLoading, onLoadMore]);

  // Only render the loader when we have more items to load
  if (!hasMore && !isLoading) return null;

  return (
    <div 
      ref={ref} 
      className="w-full py-4 flex justify-center items-center"
      aria-live="polite"
      aria-busy={isLoading}
    >
      {isLoading ? (
        <div className="flex flex-col items-center">
          <div aria-hidden="true" className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[#FE90EA]" />
          <p className={`mt-2 text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
            Loading more...
          </p>
        </div>
      ) : (
        <button
          onClick={onLoadMore}
          className={`px-4 py-2 rounded-md ${
            darkMode 
              ? "bg-gray-700 hover:bg-gray-600 text-gray-200" 
              : "bg-gray-100 hover:bg-gray-200 text-gray-700"
          } transition-colors focus:outline-none focus:ring-2 focus:ring-[#FE90EA]`}
        >
          Load more
        </button>
      )}
    </div>
  );
});

// EmptySearchResults component
const EmptySearchResults = React.memo(({ query, darkMode }) => {
  return (
    <div className={`${darkMode ? "bg-gray-800" : "bg-white"} p-6 rounded-lg shadow-sm text-center`}>
      <div className="flex flex-col items-center justify-center py-8">
        <Search size={48} className={`${darkMode ? "text-gray-600" : "text-gray-300"} mb-4`} />
        <p className={`text-lg ${darkMode ? "text-gray-300" : "text-gray-600"} mb-2`}>
          No results found for "{query}"
        </p>
        <p className={`text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
          Try a different search term or adjust your filters
        </p>
      </div>
    </div>
  );
});

// LoadingIndicator component
const LoadingIndicator = React.memo(({ darkMode }) => {
  return (
    <div className="flex justify-center items-center h-64">
      <div className="flex flex-col items-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#FE90EA]" />
        <p className={`mt-4 text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
          Searching for results...
        </p>
      </div>
    </div>
  );
});

// Optimized SellerCard with proper initialization of all variables
const SellerCard = React.memo(({ seller, darkMode, handleSellerClick, onHover, onLeave }) => {
  // Safely initialize products array
  const products = useMemo(() => seller?.products || [], [seller]);
  
  // Calculate ratings once
  const avgRating = useMemo(() => getAverageRating(products), [products]);
  
  // Sort products by image availability and score
  const prioritizedProducts = useMemo(() => 
    prioritizeValidImages(products), 
    [products]
  );
  
  // Get best product safely
  const bestProduct = useMemo(() => 
    prioritizedProducts.length > 0 ? prioritizedProducts[0] : null, 
    [prioritizedProducts]
  );
  
  // Optimize event handlers with useCallback
  const handleClick = useCallback(() => {
    if (handleSellerClick) handleSellerClick(seller.id);
  }, [handleSellerClick, seller?.id]);
  
  const handleMouseEnter = useCallback((e) => {
    if (onHover) onHover(seller, e, true);
  }, [onHover, seller]);
  
  // Use intersection observer to detect visibility
  const { ref, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true
  });
  
  // Safety check for empty seller
  if (!seller || !seller.id) {
    return null;
  }
  

  return (
    <div
      ref={ref}
      className={`${
        darkMode
          ? "bg-gray-700 hover:bg-gray-600"
          : "bg-gray-50 hover:bg-gray-100"
      } 
        rounded-lg overflow-hidden cursor-pointer shadow-sm transition-all hover:shadow-md relative product-card`}
      onClick={() => handleSellerClick(seller.id)}
      onMouseEnter={(e) => onHover && onHover(seller, e, true)}
      onMouseLeave={onLeave}
    >
      {/* Score badge - for consistency with product cards */}
      {/* <div className="absolute top-2 left-2 bg-white/90 dark:bg-gray-800/90 py-0.5 px-1.5 rounded text-xs font-medium flex items-center z-40">
        <span>
          {seller.scoreLabel || "Score"}: 
        </span>
        <span className="text-[#FE90EA] ml-1">
          {seller.displayScore || (seller.compositeScore ? seller.compositeScore.toFixed(2) : "N/A")}
        </span>
      </div> */}

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
                    src={
                      product.thumbnail_url ||
                      generatePlaceholder(300, 150, product.name)
                    }
                    alt={product.name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      e.target.src = generatePlaceholder(
                        300,
                        150,
                        product.name
                      );
                    }}
                  />
                  {/* Price tag */}
                  {product.price_cents !== undefined && (
                    <div
                      className="absolute rounded-md top-2 right-2 flex items-center"
                      style={{ zIndex: 30 }}
                    >
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
                  src={
                    prioritizedProducts[0].thumbnail_url ||
                    generatePlaceholder(400, 200, prioritizedProducts[0].name)
                  }
                  alt={prioritizedProducts[0].name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = generatePlaceholder(
                      400,
                      200,
                      prioritizedProducts[0].name
                    );
                  }}
                />
                {/* Price tag */}
                {prioritizedProducts[0].price_cents !== undefined && (
                  <div
                    className="absolute rounded-md top-2 right-2 flex items-center"
                    style={{ zIndex: 30 }}
                  >
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
                      src={
                        product.thumbnail_url ||
                        generatePlaceholder(200, 100, product.name)
                      }
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = generatePlaceholder(
                          200,
                          100,
                          product.name
                        );
                      }}
                    />
                    {/* Price tag */}
                    {product.price_cents !== undefined && (
                      <div
                        className="absolute rounded-md top-2 right-2 flex items-center"
                        style={{ zIndex: 30 }}
                      >
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
                  src={
                    prioritizedProducts[0].thumbnail_url ||
                    generatePlaceholder(300, 150, prioritizedProducts[0].name)
                  }
                  alt={prioritizedProducts[0].name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = generatePlaceholder(
                      300,
                      150,
                      prioritizedProducts[0].name
                    );
                  }}
                />
                {/* Price tag */}
                {prioritizedProducts[0].price_cents !== undefined && (
                  <div
                    className="absolute rounded-md top-2 right-2 flex items-center"
                    style={{ zIndex: 30 }}
                  >
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
                      src={
                        product.thumbnail_url ||
                        generatePlaceholder(100, 100, product.name)
                      }
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = generatePlaceholder(
                          100,
                          100,
                          product.name
                        );
                      }}
                    />
                    {/* Price tag - only show if not the last one with +N overlay */}
                    {product.price_cents !== undefined &&
                      !(idx === 2 && products.length > 4) && (
                        <div
                          className="absolute rounded-md top-1 right-1 flex items-center"
                          style={{ zIndex: 30 }}
                        >
                          <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                            ${(product.price_cents / 100).toFixed(2)}
                            <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                            <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                          </div>
                        </div>
                      )}
                    {idx === 2 && products.length > 4 && (
                      <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                        <span className="text-white text-sm font-bold">
                          +{products.length - 4}
                        </span>
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
                    src={
                      product.thumbnail_url ||
                      generatePlaceholder(400, 150, product.name)
                    }
                    alt={product.name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      e.target.src = generatePlaceholder(
                        400,
                        150,
                        product.name
                      );
                    }}
                  />
                  {/* Price tag */}
                  {product.price_cents !== undefined && (
                    <div
                      className="absolute rounded-md top-2 right-2 flex items-center"
                      style={{ zIndex: 30 }}
                    >
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
                  src={
                    prioritizedProducts[0].thumbnail_url ||
                    generatePlaceholder(400, 150, prioritizedProducts[0].name)
                  }
                  alt={prioritizedProducts[0].name}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = generatePlaceholder(
                      400,
                      150,
                      prioritizedProducts[0].name
                    );
                  }}
                />
                {/* Price tag */}
                {prioritizedProducts[0].price_cents !== undefined && (
                  <div
                    className="absolute rounded-md top-2 right-2 flex items-center"
                    style={{ zIndex: 30 }}
                  >
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
                      src={
                        product.thumbnail_url ||
                        generatePlaceholder(200, 150, product.name)
                      }
                      alt={product.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = generatePlaceholder(
                          200,
                          150,
                          product.name
                        );
                      }}
                    />
                    {/* Price tag */}
                    {product.price_cents !== undefined &&
                      !(idx === 1 && products.length > 3) && (
                        <div
                          className="absolute rounded-md top-2 right-2 flex items-center"
                          style={{ zIndex: 30 }}
                        >
                          <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-xs border border-t-transparent border-l-black border-r-transparent border-b-black">
                            ${(product.price_cents / 100).toFixed(2)}
                            <div className="absolute -right-[3px] -top-[1px] w-0 h-0 border-t-[6px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-black"></div>
                            <div className="absolute -right-[3px] bottom-[1px] w-0 h-0 border-t-[5px] border-b-[5px] border-l-[4px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                          </div>
                        </div>
                      )}
                    {idx === 1 && products.length > 3 && (
                      <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                        <span className="text-white text-lg font-bold">
                          +{products.length - 3}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Best product title overlay at the bottom with swipe animation */}
        <div
          className="absolute bottom-0 left-0 right-0 bg-black p-2 overflow-hidden"
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

        {/* Hover effect for product titles */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
          <div
            className="absolute bottom-0 left-0 right-0 p-3 backdrop-blur-sm bg-black/50"
            style={{
              opacity: 0.9,
            }}
          >
            <h3 className="font-medium text-sm text-gray-100 mb-1 line-clamp-1">
              {bestProduct && bestProduct.name}
            </h3>
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
                  e.target.src = `https://placehold.co/100x100?text=${seller.name.charAt(
                    0
                  )}`;
                }}
              />
            </div>
          )}

          <span
            className={`text-xs truncate ${
              darkMode ? "text-gray-300" : "text-gray-700"
            } mr-1 max-w-[80px]`}
          >
            {seller.name}
          </span>

          {/* Compact rating with count */}
          {avgRating && (
            <div className="flex items-center text-yellow-400 ml-auto mr-1">
              <span>★</span>
              <span
                className={`text-xs ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                {avgRating}
              </span>
              <span
                className={`text-xs ${
                  darkMode ? "text-gray-400" : "text-gray-500"
                } ml-1`}
              >
                (
                {
                  products.filter(
                    (p) => p.ratings_score != null && p.ratings_score > 0
                  ).length
                }
                )
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center">
          {/* Blue tag showing n+ for products count when there are at least 2 products */}
          {products.length >= 2 ? (
            <div className="px-1.5 py-0.5 bg-blue-500 text-white rounded-full text-xs flex-shrink-0">
              {products.length - 1}+ more
            </div>
          ) : (
            <span
              className={`text-xs ${
                darkMode ? "text-gray-400" : "text-gray-500"
              }`}
            >
              {products.length} product
            </span>
          )}
        </div>
      </div>
    </div>
  );
});

const SearchResultsWithSellerFilter = ({
  searchResults,
  darkMode,
  isLoading,
  query,
  renderProductCard,
  onHover,
  onLeave,
  displayedQuery
}) => {
  // Start performance measurement
  const renderStartTime = useRef(performance.now());
  useEffect(() => {
    renderStartTime.current = performance.now();
    
    return () => {
      const renderTime = performance.now() - renderStartTime.current;
      console.log(`SearchResults render took ${renderTime.toFixed(2)}ms`);
    };
  });
  
  // Component state
  const [groupBySeller, setGroupBySeller] = useState(false);
  const [selectedSeller, setSelectedSeller] = useState(null);
  const [sellerGroups, setSellerGroups] = useState([]);
  const [isMobile, setIsMobile] = useState(false);
  const [sortOrder, setSortOrder] = useState('score'); // 'score', 'price', 'rating'
  const [showGridView, setShowGridView] = useState(true);
  const [visibleItems, setVisibleItems] = useState(12); // Number of items initially visible
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  
  // For debug performance tracking
  const dataProcessingTimeRef = useRef(0);
  const debugRef = useRef({
    sortOrder,
    resultsBeforeSort: [],
    resultsAfterSort: []
  });
  
  // DEBUG: Log when sort order changes
  useEffect(() => {
    console.log(`%c Sort Order Changed: ${sortOrder}`, 'background: #FE90EA; color: black; padding: 2px 5px; border-radius: 3px;');
    debugRef.current.sortOrder = sortOrder;
  }, [sortOrder]);
  
  // Check for mobile viewport
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
    };
    
    handleResize(); // Set initial value
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Helper function for Bayesian average calculation
  const bayesianAverage = useCallback((averageScore, numRatings, m = 10, C = 3.0) => {
    /**
     * Calculate the Bayesian average rating for a product.
     * 
     * @param {number} averageScore - Average rating of the product (0 to 5)
     * @param {number} numRatings - Number of ratings the product has
     * @param {number} [m=10] - Minimum ratings for confidence
     * @param {number} [C=3.0] - Global average rating across all products
     * @returns {number} Bayesian-adjusted score
     * @throws {Error} If inputs are invalid
     */
    if (numRatings < 0 || averageScore < 0 || averageScore > 5) {
        throw new Error("Invalid input: numRatings must be >= 0 and averageScore must be between 0 and 5");
    }

    const numerator = (numRatings * averageScore) + (m * C);
    const denominator = numRatings + m;

    return numerator / denominator;
  }, []);
  
  // Process search results to create seller groups and apply Bayesian average to each product
  useEffect(() => {
    if (!searchResults || searchResults.length === 0) {
      setSellerGroups([]);
      return;
    }
    
    console.log(`%c Processing ${searchResults.length} search results`, 'background: #333; color: #FE90EA; padding: 2px 5px; border-radius: 3px;');
    const processingStart = performance.now();
    
    // DEBUG: Log sample of initial search results
    console.log("Sample of original search results:", searchResults.slice(0, 3).map(product => ({
      id: product.id,
      name: product.name,
      score: product.score,
      parsedScore: parseFloat(product.score || 0),
      seller: product.seller_name
    })));
    
    // Filter out products with scores below 0.55 and pre-process remaining products
    const processedResults = searchResults
      .filter(product => {
        // Skip products with no score or score below threshold
        const score = product.score ? parseFloat(product.score) : 0;
        return score >= 0.55;  // Consistently use >= for the threshold
      })
      .map(product => {
        const processedProduct = { ...product };
        
        // Ensure score is always a number
        if (processedProduct.score !== undefined) {
          processedProduct.numericScore = parseFloat(processedProduct.score);
        } else {
          processedProduct.numericScore = 0;
        }
        
        // Apply Bayesian average to product ratings if available
        if (processedProduct.ratings_score && processedProduct.ratings_count) {
          processedProduct.bayesianRating = bayesianAverage(
            processedProduct.ratings_score,
            processedProduct.ratings_count,
            10,  // m parameter - minimum ratings for confidence
            3.0  // C parameter - global average rating
          );
        } else {
          processedProduct.bayesianRating = 0;
        }
        
        // Create a more balanced combined score using Wilson score interval method
        if (processedProduct.numericScore) {
          // Wilson score-like approach for combining relevance score with ratings
          // This gives more weight to products with many ratings while still respecting original score
          const ratingWeight = processedProduct.ratings_count ? 
            Math.min(0.5, processedProduct.ratings_count / 50) : 0; // Max weight of 0.5 at 50+ ratings
            
          const relevanceWeight = 1 - ratingWeight;
          
          // Combine with weighted average, giving more influence to ratings as count increases
          processedProduct.bayesianScore = processedProduct.numericScore 
          // (processedProduct.numericScore * relevanceWeight) + 
          //   ((processedProduct.bayesianRating / 5) * ratingWeight);
            
          // Store the weights for debugging
          processedProduct.relevanceWeight = relevanceWeight;
          processedProduct.ratingWeight = ratingWeight;
        } else {
          processedProduct.bayesianScore = 0;
        }
        
        return processedProduct;
      });
    
    // DEBUG: Log processed results
    console.log("Sample of processed results:", processedResults.slice(0, 3).map(product => ({
      id: product.id,
      name: product.name,
      origScore: product.score,
      numericScore: product.numericScore,
      bayesianScore: product.bayesianScore?.toFixed(4),
      seller: product.seller_name
    })));
    
    // Create seller groups from processed search results
    const groups = {};
    
    processedResults.forEach((product) => {
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
      const validScores = seller.products.filter((p) => p.numericScore != null);
      if (validScores.length > 0) {
        // Store simple average for reference
        seller.simpleAvgScore = validScores.reduce((acc, product) => 
          acc + product.numericScore, 0) / validScores.length;
        
        // Enhanced score calculation - sum all scores instead of using average
        const totalScore = validScores.reduce((acc, product) => 
          acc + product.numericScore, 0);
        
        // Apply logarithmic bonus for product count
        const productCountBonus = Math.log10(1 + validScores.length) * 0.1;
        seller.enhancedScore = totalScore * (1 + productCountBonus) / validScores.length;
        
        // Max product score for tie-breaking
        seller.maxProductScore = Math.max(...validScores.map((p) => p.numericScore));
        
        // Composite score for sorting
        seller.compositeScore = seller.enhancedScore * 0.7 + seller.maxProductScore * 0.3;
        
        // Calculate ratings data using Bayesian average
        // Filter products with ratings
        const productsWithRatings = seller.products.filter(p => p.ratings_score > 0);
        
        // Calculate total number of ratings (not just products with ratings)
        seller.totalRatingsCount = seller.products.reduce((acc, p) => 
          acc + (p.ratings_count || 0), 0);
        
        // Calculate average rating if there are products with ratings
        if (productsWithRatings.length > 0) {
          // Calculate weighted average based on number of ratings per product
          const totalWeightedRating = productsWithRatings.reduce((acc, p) => 
            acc + (p.ratings_score * (p.ratings_count || 1)), 0);
          const totalRatingsUsed = productsWithRatings.reduce((acc, p) => 
            acc + (p.ratings_count || 1), 0);
          
          const rawAvgRating = totalWeightedRating / totalRatingsUsed;
          
          // Apply Bayesian average to the rating
          seller.avgRating = bayesianAverage(
            rawAvgRating, 
            seller.totalRatingsCount, 
            2,  // m parameter - minimum ratings for confidence
            3.0  // C parameter - global average rating
          );
          
          // Add the raw rating for reference
          seller.rawAvgRating = rawAvgRating;
          
          // Combine rating with product score for a more balanced ranking
          seller.combinedScore = (seller.compositeScore * 0.7) + 
            ((seller.avgRating / 5) * 0.3);  // Normalize rating to 0-1 scale
        } else {
          seller.avgRating = 0;
          seller.rawAvgRating = 0;
          seller.combinedScore = seller.compositeScore;
        }
        
        // Calculate average price
        const productsWithPrice = seller.products.filter(p => p.price_cents > 0);
        if (productsWithPrice.length > 0) {
          seller.avgPrice = productsWithPrice.reduce((acc, p) => acc + p.price_cents, 0) / productsWithPrice.length;
        }
      } else {
        seller.simpleAvgScore = 0;
        seller.enhancedScore = 0;
        seller.maxProductScore = 0;
        seller.compositeScore = 0;
        seller.totalRatingsCount = 0;
        seller.avgRating = 0;
        seller.rawAvgRating = 0;
        seller.combinedScore = 0;
      }
    });
    
    // DEBUG: Log sample of seller groups
    console.log("Sample of seller groups:", Object.values(groups).slice(0, 2).map(seller => ({
      id: seller.id,
      name: seller.name,
      productCount: seller.products.length,
      compositeScore: seller.compositeScore?.toFixed(4),
      combinedScore: seller.combinedScore?.toFixed(4)
    })));
    
    // Set seller groups
    setSellerGroups(Object.values(groups));
    
    // Store processing time for performance analysis
    dataProcessingTimeRef.current = performance.now() - processingStart;
    console.log(`Data processing took ${dataProcessingTimeRef.current.toFixed(2)}ms`);
  }, [searchResults, bayesianAverage]);
  
  // Handle loading more items with pagination
  const handleLoadMore = useCallback(() => {
    setIsLoadingMore(true);
    // Simulate loading delay if needed
    setTimeout(() => {
      setVisibleItems(prev => prev + 12); // Load 12 more items
      setIsLoadingMore(false);
    }, 300);
  }, []);
  
  // Memoize handlers to prevent recreations
  const handleSellerClick = useCallback((sellerId) => {
    setSelectedSeller(prev => prev === sellerId ? null : sellerId);
  }, []);
  
  const toggleGroupBySeller = useCallback(() => {
    setGroupBySeller(prev => !prev);
    setSelectedSeller(null);
  }, []);
  
  const toggleViewMode = useCallback(() => {
    setShowGridView(prev => !prev);
  }, []);
  
  const handleSortChange = useCallback((newSortOrder) => {
    console.log(`%c Sort Change: ${sortOrder} -> ${newSortOrder}`, 'background: #FE90EA; color: black; padding: 2px 5px; border-radius: 3px;');
    setSortOrder(newSortOrder);
  }, [sortOrder]);
  

  // Memoize filtered results - now filters by score threshold and selected seller
  const filteredResults = useMemo(() => {
    // First filter by minimum score threshold - use consistent comparison
    const scoreFilteredResults = searchResults.filter(product => {
      // Handle string or number scores, convert to number with parseFloat
      const score = product.score ? parseFloat(product.score) : 0;
      // Ensure we're using a consistent comparison
      return score >= 0.55;  // Consistently use >= to include 0.55 exactly
    });
    
    // DEBUG: Log filtered results count
    console.log(`Filtered ${searchResults.length} results to ${scoreFilteredResults.length}`);
    
    // Then filter by selected seller if needed
    if (!selectedSeller) return scoreFilteredResults;
    
    const sellerFiltered = scoreFilteredResults.filter(product => 
      product.seller_id === selectedSeller || 
      (product.seller_name && `seller-${product.seller_name}` === selectedSeller)
    );
    
    console.log(`Further filtered to ${sellerFiltered.length} results from seller: ${selectedSeller}`);
    return sellerFiltered;
  }, [searchResults, selectedSeller]);
  
  // Sort results based on selected sort order, now using cleaner sort functions
  const sortedResults = useMemo(() => {
    const resultsToSort = [...(selectedSeller ? filteredResults : searchResults)];
    
    // Store results before sorting for debugging
    debugRef.current.resultsBeforeSort = resultsToSort.slice(0, 5).map(product => ({
      id: product.id,
      name: product.name?.substring(0, 20),
      score: product.score,
      numericScore: parseFloat(product.score || 0),
      bayesianScore: product.bayesianScore
    }));
    
    console.log(`%c Sorting ${resultsToSort.length} results by ${sortOrder}`, 'background: #333; color: #FE90EA; padding: 2px 5px; border-radius: 3px;');
    console.log("Sample of results BEFORE sorting:", debugRef.current.resultsBeforeSort);
    
    let sorted = [];
    
    switch (sortOrder) {
      case 'price-asc':
        sorted = resultsToSort.sort((a, b) => {
          // Handle missing prices or zero prices properly
          const aPrice = a.price_cents !== undefined ? a.price_cents : Infinity;
          const bPrice = b.price_cents !== undefined ? b.price_cents : Infinity;
          
          // This ensures $0.00 items appear first when sorting by price ascending
          return aPrice - bPrice;
        });
        break;
        
      case 'price-desc':
        sorted = resultsToSort.sort((a, b) => {
          // Handle missing prices or zero prices properly
          const aPrice = a.price_cents !== undefined ? a.price_cents : -1;
          const bPrice = b.price_cents !== undefined ? b.price_cents : -1;
          
          // This ensures $0.00 items appear last when sorting by price descending
          return bPrice - aPrice;
        });
        break;
      
      case 'rating':
        sorted = resultsToSort.sort((a, b) => {
          // Use bayesianRating directly when sorting by rating
          const aBayesian = a.bayesianRating || 0;
          const bBayesian = b.bayesianRating || 0;
          
          // If ratings are identical, use a more sophisticated tie-breaker
          if (Math.abs(bBayesian - aBayesian) < 0.01) {
            // First check rating count
            const aCount = a.ratings_count || 0;
            const bCount = b.ratings_count || 0;
            
            // If counts differ by more than 5, use count as tie-breaker
            if (Math.abs(bCount - aCount) > 5) {
              return bCount - aCount;
            }
            
            // Otherwise, factor in the relevance score as a secondary tie-breaker
            return parseFloat(b.score || 0) - parseFloat(a.score || 0);
          }
          return bBayesian - aBayesian;
        });
        break;
        
      case 'score':
      default:
        // Use numeric scores for comparison to ensure consistent sorting
        sorted = resultsToSort.sort((a, b) => {
          // Convert scores to numbers for comparison
          const aScore = parseFloat(a.score || 0);
          const bScore = parseFloat(b.score || 0);
          
          // If scores are very close, use ratings as a tie-breaker
          if (Math.abs(bScore - aScore) < 0.001) {
            // Use ratings count as tie-breaker
            const aCount = a.ratings_count || 0;
            const bCount = b.ratings_count || 0;
            
            if (aCount !== bCount) {
              return bCount - aCount;
            }
            
            // If rating counts are the same, use rating score
            const aRating = a.ratings_score || 0;
            const bRating = b.ratings_score || 0;
            
            return bRating - aRating;
          }
          
          // Primary sort by score
          return bScore - aScore;
        });
        break;
    }
    
    // Store results after sorting for debugging
    debugRef.current.resultsAfterSort = sorted.slice(0, 5).map(product => ({
      id: product.id,
      name: product.name?.substring(0, 20),
      score: product.score,
      numericScore: parseFloat(product.score || 0),
      bayesianScore: product.bayesianScore
    }));
    
    console.log("Sample of results AFTER sorting:", debugRef.current.resultsAfterSort);
    return sorted;
  }, [filteredResults, searchResults, selectedSeller, sortOrder]);
  

  // Get selected seller name
  const selectedSellerName = useMemo(() => {
    if (!selectedSeller) return "";
    const seller = sellerGroups.find(s => s.id === selectedSeller);
    return seller ? seller.name : "";
  }, [selectedSeller, sellerGroups]);
  

  /**
 * Calculate the ideal DCG for normalization
 * @param {number} length - Number of items
 * @return {number} - The ideal DCG value
 */
const calculateIdealDCG = (length) => {
  return Array(length).fill(1).reduce((acc, value, index) => {
    return acc + value / Math.log2(2 + index);
  }, 0);
};

/**
 * Calculate the Normalized Discounted Cumulative Gain for a set of products
 * @param {Array} products - Array of products with score property
 * @return {number} - The NDCG score
 */
const calculateNDCG = (products) => {
  const validProducts = products.filter(p => p.score != null);
  if (validProducts.length === 0) return 0;

  const sortedProducts = [...validProducts].sort((a, b) => {
    return parseFloat(b.score) - parseFloat(a.score);
  });
  
  const dcg = sortedProducts.reduce((acc, product, index) => {
    const score = Math.max(0, parseFloat(product.score || 0));
    return acc + score / Math.log2(2 + index);
  }, 0);
  
  const idealDCG = calculateIdealDCG(validProducts.length);
  
  if (idealDCG === 0) return 0;
  
  return dcg / idealDCG;
};

/**
 *Balancing relevance vs ratings and adjusts for query relevance like an animal.
 */
const unifiedResultsWithNDCG = useMemo(() => {
  if (!searchResults || searchResults.length === 0) {
    return [];
  }

  // Start execution timer
  const startTime = performance.now();
  
  // Extract the first word of the query for special title matching
  const firstQueryWord = query.toLowerCase().split(/\s+/)[0];
  const firstQueryWordImportant = firstQueryWord && firstQueryWord.length > 2;
  
  console.log(`Creating unified results with NDCG for sorting`);
  
  try {
    let results;
    
    // Group by seller logic
    if (!groupBySeller || selectedSeller) {
      // Process individual products
      results = filteredResults.map(product => {
        // Ensure score is numeric
        const numericScore = parseFloat(product.score || 0);
        
        // Calculate rating boost - use a more efficient calculation
        const ratingBoost = (product.ratings_score && product.ratings_count) 
          ? (product.ratings_score / 5) * Math.min(0.1, product.ratings_count / 50)
          : 0;
        
        // Title match boost - only calculate when needed
        let titleMatchBoost = 0;
        if (!product.ratings_score && firstQueryWordImportant && 
            (product.name || '').toLowerCase().includes(firstQueryWord)) {
          titleMatchBoost = 0.10;
        }
        
        // Final score calculation
        const balancedScore = Math.min(numericScore + ratingBoost + titleMatchBoost, 1.0);
        
        return {
          type: 'product',
          product: {
            ...product,
            numericScore,
            originalScore: numericScore,
            ratingBoost,
            titleMatchBoost,
            scoreLabel: 'Product'
          },
          score: balancedScore,
          rating: product.ratings_score || 0,
          ratingCount: product.ratings_count || 0,
          price: product.price_cents || 0,
          originalIndex: searchResults.indexOf(product)
        };
      });
    } else {
      // Process sellers by grouping them
      const combinedResults = [];
      
      // Extract query terms for relevance evaluation
      const queryTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 2);
      
      // First, enhance seller groups with NDCG scores
      const enhancedSellerGroups = sellerGroups.map(seller => {
        if (!seller.products || seller.products.length === 0) {
          return seller;
        }
        
        // Calculate query relevance for seller's products
        let queryRelevanceScore = 0;
        if (queryTerms.length > 0) {
          // Calculate how many products from this seller match the query terms in name/description
          const productsWithQueryTerms = seller.products.filter(product => {
            const productText = `${product.name || ''} ${product.description || ''}`.toLowerCase();
            return queryTerms.some(term => productText.includes(term));
          });
          
          // Ratio of query-relevant products to total products
          queryRelevanceScore = productsWithQueryTerms.length / seller.products.length;
        }
        
        // Calculate basic metrics
        const validScores = seller.products.filter(p => p.score != null);
        const avgScore = validScores.length > 0 
          ? validScores.reduce((acc, p) => acc + parseFloat(p.score || 0), 0) / validScores.length 
          : 0;
        
        // Product count bonus - logarithmic to prevent too much weight for large catalogs
        const productCountBonus = Math.log10(1 + seller.products.length) * 0.05;
        
        // Calculate rating factor if available
        const ratingFactor = seller.avgRating 
          ? (seller.avgRating / 5) * (Math.min(0.5, seller.totalRatingsCount / 50)) * 0.15
          : 0;
        
        // Combine all factors for a balanced score
        // Weights: 60% avg score, 10% query relevance, 5% product count, 5% ratings
        const enhancedScore = (avgScore * 0.4) + 
                              (queryRelevanceScore * 0.1) +
                              (productCountBonus * 0.2) + 
                              (ratingFactor * 0.3);
        
        return {
          ...seller,
          enhancedScore,
          avgScore,
          queryRelevanceScore
        };
      });
      
      // Find max scores to normalize
      let maxProductScore = 0;
      let maxSellerScore = 0;
      
      // Find max product score
      filteredResults.forEach(product => {
        const score = parseFloat(product.score || 0);
        if (score > maxProductScore) maxProductScore = score;
      });
      
      // Find max seller score
      enhancedSellerGroups.forEach(seller => {
        if (seller.enhancedScore > maxSellerScore) maxSellerScore = seller.enhancedScore;
      });
      
      // Process the enhanced seller groups with normalized scores
      enhancedSellerGroups.forEach((seller, index) => {
        if (seller.products && seller.products.length > 0) {
          // Normalize seller scores with scaling factor
          let normalizedScore = seller.enhancedScore;
          
          if (seller.products.length === 1 && renderProductCard) {
            // For sellers with only one product, add as a product
            const product = seller.products[0];
            
            combinedResults.push({
              type: 'product',
              product: {
                ...product,
                displayScore: normalizedScore.toFixed(4),
                numericScore: normalizedScore || parseFloat(product.score || 0),
                scoreLabel: 'Product'
              },
              score: normalizedScore || parseFloat(product.score || 0),
              rating: seller.avgRating || (product.ratings_score || 0),
              ratingCount: seller.totalRatingsCount || (product.ratings_count || 0),
              price: product.price_cents || Infinity,
              originalIndex: index
            });
          } else {
            // For sellers with multiple products, add as a seller with normalized score
            combinedResults.push({
              type: 'seller',
              seller: {
                ...seller,
                displayScore: normalizedScore.toFixed(4),
                avgScoreDisplay: seller.avgScore.toFixed(4),
                scoreLabel: 'Seller'
              },
              score: normalizedScore || 0,
              rating: seller.avgRating || 0,
              ratingCount: seller.totalRatingsCount || 0,
              price: seller.avgPrice || Infinity,
              productCount: seller.products.length,
              originalIndex: index
            });
          }
        }
      });
      
      results = combinedResults;
    }
    
    // Log performance
    const endTime = performance.now();
    console.log(`Generated ${results ? results.length : 0} unified results in ${(endTime - startTime).toFixed(1)}ms`);
    
    return results || [];
  } catch (error) {
    console.error('Error generating unified results:', error);
    return [];
  }
}, [filteredResults, groupBySeller, selectedSeller, sellerGroups, searchResults, query, renderProductCard]);



const debugScoreLogger = (sortedResults) => {
  console.log("\n======= SORTED SCORES DEBUG OUTPUT =======");
  console.log("Type\tRating\tScore\tNumericScore\tDisplayScore\tNormalizedScore\tIndex\tName");
  console.log("----------------------------------------------------------------------------------------------------");
  
  sortedResults.forEach((item, index) => {
    const type = item.type === 'product' ? 'PRODUCT' : 'SELLER';
    
    // Get rating value
    const rating = item.rating || 0;
    
    // Get various score values
    const score = item.score ? item.score.toFixed(4) : '0.0000';
    
    // Get numeric score (raw score value)
    const numericScore = item.type === 'product' 
      ? (item.product.numericScore || 0).toFixed(4)
      : (item.seller.avgScore || 0).toFixed(4);
    
    // Get display score
    const displayScore = item.type === 'product'
      ? item.product.displayScore || 'N/A'
      : item.seller.displayScore || 'N/A';
    
    // Get normalized score (for comparison)
    const normalizedScore = item.score ? item.score.toFixed(4) : '0.0000';
    
    // Get name (truncated)
    const name = item.type === 'product'
      ? (item.product.name || '').substring(0, 50)
      : (item.seller.name || '').substring(0, 50);
    
    // Print row
    console.log(`${type}\t${rating.toFixed(1)}\t${score}\t${numericScore}\t${displayScore}\t${normalizedScore}\t${index}\t${name}`);
  });
  
  console.log("======= END DEBUG OUTPUT =======\n");
};

const sortedUnifiedResultsWithNDCG = useMemo(() => {
  // Debug starting sort
  console.log(`%c Sorting unified results with NDCG by ${sortOrder}`, 'background: #333; color: #FE90EA; padding: 2px 5px; border-radius: 3px;');
  
  // Print the first few items before sorting for reference
  console.log("First 3 items BEFORE sorting:");
  unifiedResultsWithNDCG.slice(0, 3).forEach((item, i) => {
    const name = item.type === 'product' ? item.product.name?.substring(0, 20) : item.seller.name;
    console.log(`${i}. ${item.type} "${name}": score=${item.score.toFixed(4)}`);
  });
  
  // Create sorted copy with debug logging
  const sorted = [...unifiedResultsWithNDCG].sort((a, b) => {
    switch (sortOrder) {
      case 'price-asc':
        // Primary sort by price ascending
        const aPrice = a.price !== undefined ? a.price : 0;
        const bPrice = b.price !== undefined ? b.price : 0;
        
        if (aPrice !== bPrice) {
          return aPrice - bPrice;
        }
        
        // Tie-breaker: score descending
        return b.score - a.score;
      case 'price-desc':
        // Primary sort by price descending
        const aPriceDesc = a.price !== undefined ? a.price : -1;
        const bPriceDesc = b.price !== undefined ? b.price : -1;
        
        if (aPriceDesc !== bPriceDesc) {
          return bPriceDesc - aPriceDesc;
        }
        
        // Tie-breaker: score descending
        return b.score - a.score;
        
      case 'rating':
        // Primary sort by rating
        if (Math.abs(b.rating - a.rating) > 0.01) {
          return b.rating - a.rating;
        }
        
        // First tie-breaker: rating count
        if (a.ratingCount !== b.ratingCount) {
          return b.ratingCount - a.ratingCount;
        }
        
        // Second tie-breaker: score
        if (Math.abs(b.score - a.score) > 0.001) {
          return b.score - a.score;
        }
        
        // Final tie-breaker: original index for stable sort
        return a.originalIndex - b.originalIndex;
        
      case 'score':
      default:
        // Primary sort by score - use a small threshold to avoid floating point issues
        if (Math.abs(b.score - a.score) > 0.001) {
          return b.score - a.score;
        }
        
        // First tie-breaker: rating
        if (Math.abs(b.rating - a.rating) > 0.01) {
          return b.rating - a.rating;
        }
        
        // Second tie-breaker: rating count
        if (a.ratingCount !== b.ratingCount) {
          return b.ratingCount - a.ratingCount;
        }
        
        // Final tie-breaker: original index for stable sort
        return a.originalIndex - b.originalIndex;
    }
  });
  
  // Print the sorted results in a detailed table format
  debugScoreLogger(sorted);
  
  return sorted;
}, [unifiedResultsWithNDCG, sortOrder]);

// Apply pagination to the sorted unified results
const paginatedUnifiedResultsWithNDCG = useMemo(() => {
  const results = sortedUnifiedResultsWithNDCG.slice(0, visibleItems);
  console.log(`Showing ${results.length} of ${sortedUnifiedResultsWithNDCG.length} unified results with NDCG`);
  return results;
}, [sortedUnifiedResultsWithNDCG, visibleItems]);

// Rendering function for the NDCG-enhanced results
const renderUnifiedWithNDCG = () => {
  // Early return with empty state message if no results
  if (paginatedUnifiedResultsWithNDCG.length === 0) {
    return (
      <EmptySearchResults query={query} darkMode={darkMode} />
    );
  }
  
  // Determine component to render based on view mode
  return (
    <div className={showGridView ? "grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4" : "space-y-4"}>
      {/* Use React.Fragment grouping with key to improve reconciliation */}
      {paginatedUnifiedResultsWithNDCG.map((item, index) => (
        <React.Fragment key={`${item.type}-${item.type === 'product' ? item.product.id || index : item.seller.id || index}`}>
          {item.type === 'product' ? (
            // Product rendering with memoization hint
            <div className={showGridView ? "" : "w-full"}>
              {renderProductCard({
                ...item.product,
                displayScore: item.score.toFixed(4)
              }, index, showGridView ? 'grid' : 'list')}
            </div>
          ) : (
            // Seller card rendering
            <SellerCard
              seller={{
                ...item.seller,
                displayScore: item.score.toFixed(4)
              }}
              darkMode={darkMode}
              handleSellerClick={handleSellerClick}
              onHover={onHover}
              onLeave={onLeave}
            />
          )}
        </React.Fragment>
      ))}
    </div>
  );
};


  // ========================


  // LoadMore component for infinite scrolling
  const LoadMore = React.memo(({ onLoadMore, hasMore, isLoading }) => {
    const { ref, inView } = useInView({
      threshold: 0.1,
      triggerOnce: false
    });

    useEffect(() => {
      if (inView && hasMore && !isLoading) {
        onLoadMore();
      }
    }, [inView, hasMore, isLoading, onLoadMore]);

    return (
      <div 
        ref={ref} 
        className="w-full py-6 flex justify-center items-center"
      >
        {isLoading ? (
          <div className="flex flex-col items-center">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[#FE90EA]" />
            <p className={`mt-2 text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
              Loading more...
            </p>
          </div>
        ) : hasMore ? (
          <button
            onClick={onLoadMore}
            className={`px-4 py-2 rounded-md ${
              darkMode 
                ? "bg-gray-700 hover:bg-gray-600 text-gray-200" 
                : "bg-gray-100 hover:bg-gray-200 text-gray-700"
            } transition-colors`}
          >
            Load more
          </button>
        ) : (
          <p className={`text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
            No more results to load
          </p>
        )}
      </div>
    );
  });
  
  
  // Empty state
  if (isLoading && (!searchResults || searchResults.length === 0)) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#FE90EA]" />
          <p className={`mt-4 text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
            Searching for results...
          </p>
        </div>
      </div>
    );
  }
  
  // Empty results
  if (!searchResults || searchResults.length === 0 || sortedResults.length === 0) {
    return (
      <div className={`${darkMode ? "bg-gray-800" : "bg-white"} p-6 rounded-lg shadow-sm text-center`}>
        <div className="flex flex-col items-center justify-center py-8">
          <Search size={48} className={`${darkMode ? "text-gray-600" : "text-gray-300"} mb-4`} />
          <p className={`text-lg ${darkMode ? "text-gray-300" : "text-gray-600"} mb-2`}>
            {searchResults && searchResults.length > 0 
              ? `No results with score of 0.55 or higher found for "${query}"`
              : `No results found for "${query}"`}
          </p>
          <p className={`text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
            Try a different search term or adjust your filters
          </p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      {/* Results header and controls */}
      <div className={`${darkMode ? "bg-gray-800" : "bg-white"} p-4 rounded-lg shadow-sm`}>
        <div className="flex flex-wrap justify-between items-center">
        <div className="flex items-center space-x-2 mb-2 sm:mb-0">
        <h2 className={`text-lg sm:text-xl font-semibold ${darkMode ? "text-white" : "text-black"} border-b-2 border-[#FE90EA] pb-2 inline-block`}>
          {selectedSeller
            ? `${selectedSellerName.split(" ")[0]}'s Products (${filteredResults.length})`
            : `Search Results (${filteredResults.length})`}
        </h2>
        <span className={`text-sm ${darkMode ? "text-gray-300" : "text-gray-600"}`}>
          related to "{displayedQuery}"
        </span>
      </div>

          <div className="flex items-center space-x-4 mt-2 sm:mt-0">
            {selectedSeller && (
              <button
                onClick={() => setSelectedSeller(null)}
                className="text-sm text-[#FE90EA] hover:underline"
              >
                View all results
              </button>
            )}

            {/* Sort options dropdown */}
            <div className="relative">
              <select
                className={`text-sm px-2 py-1 pr-8 rounded-md border ${
                  darkMode 
                    ? 'bg-gray-700 border-gray-600 text-gray-300' 
                    : 'bg-white border-gray-300 text-gray-700'
                } focus:outline-none focus:ring-1 focus:ring-[#FE90EA]`}
                value={sortOrder}
                onChange={(e) => handleSortChange(e.target.value)}
              >
                <option value="score">Sort by Relevance</option>
                <option value="rating">Sort by Rating</option>
                <option value="price-asc">Price: Low to High</option>
                <option value="price-desc">Price: High to Low</option>
              </select>
            </div>

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

      <div className={`${darkMode ? "bg-gray-800" : "bg-white"} p-4 rounded-lg shadow-sm`}>
      {isLoading && searchResults.length > 0 ? (
  <LoadingIndicator darkMode={darkMode} />
) : (
  <div className="min-h-[200px]">
    {renderUnifiedWithNDCG()}
    
    {/* Load more trigger and indicator */}
    <OptimizedLoadMore 
      onLoadMore={handleLoadMore}
      hasMore={paginatedUnifiedResultsWithNDCG.length < sortedUnifiedResultsWithNDCG.length}
      isLoading={isLoadingMore}
      darkMode={darkMode}
    />
  </div>
)}
      </div>
    </div>
  );
};

export default React.memo(SearchResultsWithSellerFilter);