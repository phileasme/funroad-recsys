import React, { useState, useEffect, useMemo, useCallback, useRef, Suspense } from 'react';
import { ChevronDown, ChevronUp, Filter, Users, Grid, List, Search } from 'lucide-react';
import { FixedSizeGrid as FGrid } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import memoize from 'memoize-one';
import { useInView } from 'react-intersection-observer';

// Performance measurements
const perfMetrics = {
  renderCount: 0,
  renderTime: 0,
  lastRenderStart: 0
};

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

function bayesianAverage(averageScore, numRatings, m = 2, C = 3.0) {
  /**
   * Calculate the Bayesian average rating for a product.
   * 
   * @param {number} averageScore - Average rating of the product (0 to 5)
   * @param {number} numRatings - Number of ratings the product has
   * @param {number} [m=10] - Minimum number of ratings for confidence
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
}

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

// Multiple Seller Cards grid component
const SellerCards = React.memo(({ 
  sellers, 
  darkMode, 
  handleSellerClick, 
  renderProductCard,
  onHover, 
  onLeave
}) => {
  // If there's a seller with only one product, render the product card directly
  const renderedSellers = useMemo(() => {
    return sellers.map((seller) => {
      if (seller.products?.length === 1 && renderProductCard) {
        const product = seller.products[0];
        // Add composite score to product for consistent ranking
        product.displayScore = seller.compositeScore || product.score;
        
        return (
          <div key={`single-${seller.id}`} className="seller-card-wrapper">
            {renderProductCard(product, 0)}
          </div>
        );
      }
      
      // Otherwise render a seller card
      return (
        <SellerCard
          key={`seller-${seller.id}`}
          seller={seller}
          darkMode={darkMode}
          handleSellerClick={handleSellerClick}
          onHover={onHover}
          onLeave={onLeave}
        />
      );
    });
  }, [sellers, darkMode, handleSellerClick, renderProductCard, onHover, onLeave]);
  
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
      {renderedSellers}
    </div>
  );
});

// LoadMore component for infinite scrolling
const LoadMore = React.memo(({ onLoadMore, hasMore, isLoading, darkMode }) => {
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

const SearchResultsWithSellerFilter = ({
  searchResults,
  darkMode,
  isLoading,
  query,
  renderProductCard,
  onHover,
  onLeave
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
          processedProduct.bayesianScore = (processedProduct.numericScore * relevanceWeight) + 
            ((processedProduct.bayesianRating / 5) * ratingWeight);
            
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
            10,  // m parameter - minimum ratings for confidence
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
  
  // Apply pagination to the sorted results
  const paginatedResults = useMemo(() => {
    const results = sortedResults.slice(0, visibleItems);
    console.log(`Showing ${results.length} of ${sortedResults.length} total results (pagination limit: ${visibleItems})`);
    return results;
  }, [sortedResults, visibleItems]);
  
  // Sorted seller groups - filtered to only include sellers with products above threshold
  const sortedSellerGroups = useMemo(() => {
    if (!sellerGroups.length) return [];
    
    // Filter out seller groups that have no products after applying score threshold
    const validSellerGroups = sellerGroups.filter(seller => {
      // Check if seller has any products after applying the score threshold
      const validProducts = seller.products.filter(product => {
        const score = product.score ? parseFloat(product.score) : 0;
        return score >= 0.55;  // Consistently use >= to include 0.55 exactly
      });
      
      // Keep only sellers with at least one valid product
      return validProducts.length > 0;
    });
    
    // DEBUG: Log before sorting sellers
    console.log(`Sorting ${validSellerGroups.length} seller groups by ${sortOrder}`);
    console.log("Sample of sellers BEFORE sorting:", validSellerGroups.slice(0, 2).map(seller => ({
      id: seller.id,
      name: seller.name,
      compositeScore: seller.compositeScore?.toFixed(4),
      combinedScore: seller.combinedScore?.toFixed(4),
      avgRating: seller.avgRating?.toFixed(2),
      avgPrice: seller.avgPrice
    })));
    
    // First clean and normalize all scores for consistent sorting
    const normalizedSellerGroups = validSellerGroups.map(seller => {
      // Make a copy to avoid mutating the original
      const normalizedSeller = {...seller};
      
      // Filter products to only include those above threshold
      normalizedSeller.products = normalizedSeller.products.filter(product => {
        const score = product.score ? parseFloat(product.score) : 0;
        return score >= 0.55;  // Consistently use >= to include 0.55 exactly
      });
      
      // Ensure consistent score types and precision
      if (normalizedSeller.compositeScore) {
        normalizedSeller.compositeScore = parseFloat(normalizedSeller.compositeScore.toFixed(4));
      }
      
      if (normalizedSeller.combinedScore) {
        normalizedSeller.combinedScore = parseFloat(normalizedSeller.combinedScore.toFixed(4));
      }
      
      // For single-product sellers, align product score with seller score
      if (normalizedSeller.products?.length === 1 && normalizedSeller.combinedScore) {
        const product = normalizedSeller.products[0];
        product.normalizedScore = normalizedSeller.combinedScore;
        
        // When in score sort mode, use the combined score (with ratings)
        if (sortOrder === 'score') {
          product.score = normalizedSeller.combinedScore.toString();
        }
      }
      
      return normalizedSeller;
    });


    // Then sort with normalized scores
    const sortedSellers = [...normalizedSellerGroups].sort((a, b) => {
      switch (sortOrder) {
        case 'price-asc':
          // Sort by average price ascending
          const aAvg = a.avgPrice !== undefined ? a.avgPrice : Number.MAX_SAFE_INTEGER;
          const bAvg = b.avgPrice !== undefined ? b.avgPrice : Number.MAX_SAFE_INTEGER;
          
          // This ensures $0.00 items appear first when sorting by price ascending
          if (aAvg !== bAvg) return aAvg - bAvg;
          
          // If prices are equal, use secondary sorting criteria
          return (a.products?.length || 0) - (b.products?.length || 0);

        case 'price-desc':
          // Sort by average price descending
          const aAvgD = a.avgPrice !== undefined ? a.avgPrice : Number.MIN_SAFE_INTEGER;
          const bAvgD = b.avgPrice !== undefined ? b.avgPrice : Number.MIN_SAFE_INTEGER;
          
          if (aAvgD !== bAvgD) return bAvgD - aAvgD;
          
          // If prices are equal, use secondary sorting criteria
          return (b.products?.length || 0) - (a.products?.length || 0);

        case 'rating':
          // Sort by Bayesian-adjusted rating
          const aRating = a.avgRating || 0;
          const bRating = b.avgRating || 0;
          
          if (Math.abs(bRating - aRating) > 0.01) {
            return bRating - aRating;
          }
          
          // If ratings are very close, use rating count as tie-breaker
          const aRatingCount = a.totalRatingsCount || 0;
          const bRatingCount = b.totalRatingsCount || 0;
          
          if (aRatingCount !== bRatingCount) {
            return bRatingCount - aRatingCount;
          }
          
          // If rating counts are equal, use product count
          return (b.products?.length || 0) - (a.products?.length || 0);

        case 'score':
        default:
          // Prioritize combined score
          const aCombinedScore = a.combinedScore || 0;
          const bCombinedScore = b.combinedScore || 0;
          
          if (Math.abs(bCombinedScore - aCombinedScore) > 0.0001) {
            return bCombinedScore - aCombinedScore;
          }
          
          // If combined scores are very close, use multiple tie-breakers
          // First, use product count
          if (a.products?.length !== b.products?.length) {
            return (b.products?.length || 0) - (a.products?.length || 0);
          }
          
          // Then, use average rating as a secondary tie-breaker
          const aSecondaryRating = a.avgRating || 0;
          const bSecondaryRating = b.avgRating || 0;
          
          if (Math.abs(bSecondaryRating - aSecondaryRating) > 0.01) {
            return bSecondaryRating - aSecondaryRating;
          }
          
          // Final tie-breaker: total ratings count
          return (b.totalRatingsCount || 0) - (a.totalRatingsCount || 0);
      }
    });
    // Then sort with normalized scores
    // const sortedSellers = [...normalizedSellerGroups].sort((a, b) => {
    //   switch (sortOrder) {
    //     case 'price-asc':
    //       // Sort by average price ascending

    //       const aAvg = a.avgPrice !== undefined ? a.avgPrice : Infinity;
    //       const bAvg = b.avgPrice !== undefined ? b.avgPrice : Infinity;
          
    //       // This ensures $0.00 items appear first when sorting by price ascending
    //       return aAvg - bAvg;
    //     case 'price-desc':
    //       // Sort by average price descending

    //       const aAvgD = a.avgPrice !== undefined ? a.avgPrice : -1;
    //       const bAvgD = b.avgPrice !== undefined ? b.avgPrice : -1;
    //       return bAvgD - aAvgD;
    //     case 'rating':
    //       // Sort by Bayesian-adjusted rating, with tie-breaker on rating count
    //       if (Math.abs((b.avgRating || 0) - (a.avgRating || 0)) < 0.01) {
    //         // If ratings are very close, use total ratings count as tie-breaker
    //         return (b.totalRatingsCount || 0) - (a.totalRatingsCount || 0);
    //       }
    //       return (b.avgRating || 0) - (a.avgRating || 0);
    //     case 'score':
    //     default:
    //       // Sort by combined score (product score + ratings) with precise comparison
    //       if (Math.abs((b.combinedScore || 0) - (a.combinedScore || 0)) < 0.0001) {
    //         // If scores are very close, use product count as tie-breaker
    //         return (b.products?.length || 0) - (a.products?.length || 0);
    //       }
    //       return (b.combinedScore || 0) - (a.combinedScore || 0);
    //   }
    // });
    
    // DEBUG: Log after sorting sellers
    console.log("Sample of sellers AFTER sorting:", sortedSellers.slice(0, 2).map(seller => ({
      id: seller.id,
      name: seller.name,
      compositeScore: seller.compositeScore?.toFixed(4),
      combinedScore: seller.combinedScore?.toFixed(4),
      avgRating: seller.avgRating?.toFixed(2),
      avgPrice: seller.avgPrice
    })));
    
    return sortedSellers;
  }, [sellerGroups, sortOrder]);
  
  // Get selected seller name
  const selectedSellerName = useMemo(() => {
    if (!selectedSeller) return "";
    const seller = sellerGroups.find(s => s.id === selectedSeller);
    return seller ? seller.name : "";
  }, [selectedSeller, sellerGroups]);
  
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
            ? `${selectedSellerName}'s Products (${filteredResults.length})`
            : `Search Results (${filteredResults.length} of ${searchResults.length})`}
        </h2>
        <span className={`text-sm ${darkMode ? "text-gray-300" : "text-gray-600"}`}>
          related to "{query}"
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

      {/* Results grid - no fixed height, uses natural document scroll */}
      <div className={`${darkMode ? "bg-gray-800" : "bg-white"} p-4 rounded-lg shadow-sm`}>
        {isLoading && searchResults.length > 0 ? (
          <div className="flex justify-center items-center h-64">
            <div className="flex flex-col items-center">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#FE90EA]" />
              <p className={`mt-4 text-sm ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
                Searching for results...
              </p>
            </div>
          </div>
        ) : (
          <div className="min-h-[200px]">
            {selectedSeller || !groupBySeller ? (
              showGridView ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                  {paginatedResults.map((product, index) => (
                    <div key={`grid-${product.id || index}`}>
                      {renderProductCard(product, index)}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-4">
                  {paginatedResults.map((product, index) => (
                    <div key={`list-${product.id || index}`} className="w-full">
                      {renderProductCard(product, index, 'list')}
                    </div>
                  ))}
                </div>
              )
            ) : (
              <SellerCards
                sellers={sortedSellerGroups}
                darkMode={darkMode}
                handleSellerClick={handleSellerClick}
                renderProductCard={renderProductCard}
                onHover={onHover}
                onLeave={onLeave}
              />
            )}
            
            {/* Load more trigger and indicator */}
            {(selectedSeller || !groupBySeller) && (
              <LoadMore 
                onLoadMore={handleLoadMore}
                hasMore={paginatedResults.length < sortedResults.length}
                isLoading={isLoadingMore}
                darkMode={darkMode}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default React.memo(SearchResultsWithSellerFilter);