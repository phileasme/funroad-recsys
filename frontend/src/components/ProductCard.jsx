import React, { useState, useCallback, useMemo } from 'react';
import OptimizedImage from './OptimizedImage';

const ProductCard = ({ product, index, darkMode, onHover, onLeave }) => {
  // State for interactions and view modes
  const [showImageView, setShowImageView] = useState(false);
  const [showFullImage, setShowFullImage] = useState(false);

  // Memoize image URLs to prevent unnecessary recalculations
  const imageUrl = useMemo(() => (
    product.proxied_thumbnail_url || product.thumbnail_url || null
  ), [product.proxied_thumbnail_url, product.thumbnail_url]);
  
  const fallbackUrl = useMemo(() => (
    product.fallback_url || `https://placehold.co/600x400/fe90ea/ffffff?text=${encodeURIComponent(product.name || 'Product')}`
  ), [product.fallback_url, product.name]);

  // Determine if this is a priority image (first few in the list)
  const isPriority = index < 5;

  // Event handlers with useCallback for better performance
  const handleMouseEnter = useCallback((e) => {
    if (onHover) onHover(product, e);
  }, [product, onHover]);

  const handleMouseLeave = useCallback(() => {
    if (onLeave) onLeave();
  }, [onLeave]);

  const toggleView = useCallback(() => {
    setShowImageView(prev => !prev);
  }, []);

  const handleImageClick = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setShowFullImage(true);
  }, []);

  const handleCloseFullImage = useCallback(() => {
    setShowFullImage(false);
  }, []);

  // Format price for display
  const formattedPrice = useMemo(() => {
    if (product.price_cents !== undefined) {
      return `$${(product.price_cents / 100).toFixed(2)}`;
    }
    return null;
  }, [product.price_cents]);

  // Seller info component
  const SellerInfo = () => {
    if (!product.seller_name && !product.seller_id) return null;
    
    return (
      <div className="flex items-center">
        <div className={`w-5 h-5 rounded-full overflow-hidden flex items-center justify-center mr-1 ${
          darkMode ? 'bg-gray-600' : 'bg-gray-200'
        }`}>
          <div 
            className="w-full h-full flex items-center justify-center text-xs font-bold"
            style={{ backgroundColor: '#FE90EA', color: 'white' }}
          >
            {product.seller_name ? product.seller_name.charAt(0).toUpperCase() : 'S'}
          </div>
        </div>
        
        {product.seller_name && (
          <span className={`text-xs ${darkMode ? 'text-gray-300' : 'text-gray-600'} truncate max-w-[80px]`}>
            {product.seller_name}
          </span>
        )}
      </div>
    );
  };

  return (
    <>
      <div 
        className={`${darkMode ? 'bg-gray-700 border-gray-600 hover:border-[#FE90EA]' : 'bg-white border-gray-200 hover:border-[#FE90EA]'} border-2 rounded-lg overflow-hidden hover:shadow-lg transition-shadow product-card h-[300px]`}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={() => window.open(product.url || "#", "_blank")}
      >
        {/* Image section with OptimizedImage component */}
        <div className={`relative ${showImageView ? 'h-full' : 'h-40'} transition-all duration-300`}>
          <OptimizedImage 
            src={imageUrl}
            alt={product.name}
            fallbackSrc={fallbackUrl}
            priority={isPriority}
            objectFit={showImageView ? 'contain' : 'cover'}
            style={{ 
              width: '100%',
              height: '100%',
              backgroundColor: darkMode ? '#2D3748' : '#F7FAFC'
            }}
            onClick={handleImageClick}
          />
          
          {/* Toggle view button */}
          <button
            className={`absolute top-2 left-2 ${darkMode ? 'bg-gray-800/70' : 'bg-white/70'} 
            p-1 rounded-full cursor-pointer hover:bg-[#FE90EA]/70 transition-colors z-30`}
            onClick={(e) => {
              e.stopPropagation();
              toggleView();
            }}
            title={showImageView ? "Show details" : "Show image"}
          >
            <svg 
              width="16" 
              height="16" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            >
              {showImageView ? (
                <>
                  <line x1="8" y1="6" x2="21" y2="6"></line>
                  <line x1="8" y1="12" x2="21" y2="12"></line>
                  <line x1="8" y1="18" x2="21" y2="18"></line>
                  <line x1="3" y1="6" x2="3.01" y2="6"></line>
                  <line x1="3" y1="12" x2="3.01" y2="12"></line>
                  <line x1="3" y1="18" x2="3.01" y2="18"></line>
                </>
              ) : (
                <>
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <circle cx="8.5" cy="8.5" r="1.5"></circle>
                  <polyline points="21 15 16 10 5 21"></polyline>
                </>
              )}
            </svg>
          </button>
          
          {/* Price tag */}
          {formattedPrice && (
            <div className="absolute rounded-md top-3 right-3 flex items-center z-30">
              <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-base border border-t-transparent border-l-black border-r-transparent border-b-black">
                {formattedPrice}
                <div className="absolute -right-[4px] -top-[1px] w-0 h-0 border-t-[8px] border-b-[7px] border-l-[5px] border-t-transparent border-b-transparent border-l-black"></div>
                <div className="absolute -right-[4px] bottom-[1px] w-0 h-0 border-t-[7px] border-b-[7px] border-l-[5px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
              </div>
            </div>
          )}
        </div>
        
        {/* Standard details section */}
        {!showImageView && (
          <div className="p-3">
            <h3 className={`font-medium text-sm ${darkMode ? 'text-gray-100' : 'text-gray-800'} mb-1 line-clamp-1`}>
              {product.name}
            </h3>
            
            {/* Rating display */}
            {product.ratings_count > 0 && product.ratings_score !== undefined && (
              <div className="flex items-center mb-1">
                <div className="flex text-yellow-400">
                  {[...Array(5)].map((_, i) => (
                    <span key={i} className="text-xs">
                      {i < Math.floor(product.ratings_score) ? "★" : "☆"}
                    </span>
                  ))}
                </div>
                <span className={`ml-1 text-xs ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  {product.ratings_score} {product.ratings_count > 0 ? `(${product.ratings_count})` : ''}
                </span>
              </div>
            )}
            
            <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} text-xs mb-2 line-clamp-2`}>
              {product.description || "No description available."}
            </p>
            
            <div className="flex items-center justify-between mt-auto pt-1 border-t border-gray-100">
              <SellerInfo />
              
              <a 
                href={product.url || "#"} 
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center px-2 py-1 text-xs font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-1 focus:ring-[#FE90EA] border border-black"
                onClick={(e) => e.stopPropagation()}
              >
                View details
              </a>
            </div>
          </div>
        )}
        
        {/* Overlay info for image view mode */}
        {showImageView && (
          <div className="absolute bottom-0 left-0 right-0 bg-black/75 p-3 z-10">
            <h3 className={`font-medium text-sm text-white mb-1 line-clamp-1`}>
              {product.name}
            </h3>
            
            <div className="flex items-center justify-between">
              <SellerInfo />
              
              <a 
                href={product.url || "#"} 
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center px-2 py-1 text-xs font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-1 focus:ring-[#FE90EA] border border-black"
                onClick={(e) => e.stopPropagation()}
              >
                View details
              </a>
            </div>
          </div>
        )}
      </div>
      
      {/* Fullscreen image view */}
      {showFullImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-90"
          onClick={handleCloseFullImage}
        >
          <div 
            className="relative max-w-4xl max-h-[85vh] w-full h-full flex items-center justify-center"
            onClick={(e) => e.stopPropagation()}
          >
            <button 
              className="absolute top-4 right-4 z-10 w-8 h-8 rounded-full bg-black bg-opacity-50 flex items-center justify-center text-white hover:bg-opacity-70 transition-colors"
              onClick={handleCloseFullImage}
              aria-label="Close fullscreen view"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
            
            <OptimizedImage
              src={imageUrl}
              alt={product.name}
              fallbackSrc={fallbackUrl}
              priority={true}
              objectFit="contain"
              className="max-w-full max-h-full cursor-pointer"
              onClick={() => window.open(product.url || "#", "_blank")}
            />
            
            <div className="absolute bottom-4 left-0 right-0 text-center text-white bg-black bg-opacity-30 py-2 px-4">
              <h3 className="font-bold text-base">{product.name}</h3>
              {product.seller_name && (
                <div className="flex items-center justify-center mt-1">
                  <div className="w-5 h-5 rounded-full overflow-hidden mr-1 bg-[#FE90EA] flex items-center justify-center">
                    <span className="text-xs font-bold text-white">
                      {product.seller_name.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <span className="text-sm opacity-90">{product.seller_name}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default React.memo(ProductCard);