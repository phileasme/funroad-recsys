import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createFallbackImageUrl, getProxiedImageUrl } from '../services/imageService';

function ProductCard({ product, index, darkMode, onHover, onLeave }) {
  // State to track hover
  const [isHovered, setIsHovered] = useState(false);
  // State to control actual display of details (with delayed timing)
  const [showDetails, setShowDetails] = useState(true);
  // State to control fullscreen image view
  const [showFullImage, setShowFullImage] = useState(false);
  // State to track if mouse is over image specifically
  const [mouseOverImage, setMouseOverImage] = useState(false);
  // State to track image aspect ratio
  const [isWideImage, setIsWideImage] = useState(false);
  // State to track window width for responsive behavior
  const [isMobile, setIsMobile] = useState(false);
  // State to control if standard description should be shown during hover
  const [showStandardOnHover, setShowStandardOnHover] = useState(false);
  // State to track image loading
  const [imageLoaded, setImageLoaded] = useState(false);
  // State to track image error
  const [imageError, setImageError] = useState(false);
  // State for image dimensions
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  // State for the image URL with fallback mechanism
  const [imageUrl, setImageUrl] = useState(
    product.proxied_thumbnail_url || 
    (product.thumbnail_url ? getProxiedImageUrl(product.thumbnail_url) : createFallbackImageUrl(product.name))
  );
  
  // State for seller thumbnail with fallback
  const [sellerThumbnailUrl, setSellerThumbnailUrl] = useState(
    product.seller_thumbnail ? getProxiedImageUrl(product.seller_thumbnail) : null
  );
  
  // State to track seller image loading
  const [sellerImageLoaded, setSellerImageLoaded] = useState(false);
  const [sellerImageError, setSellerImageError] = useState(false);
  
  // Refs for height calculation and timeout management
  const cardRef = useRef(null);
  const timeoutRef = useRef(null);
  const imageRef = useRef(null);
  const resizeHandlerRef = useRef(null);
  
  // Store the original height of the card
  const [cardHeight, setCardHeight] = useState(null);
  
  // Fallback image URL as a backup
  const fallbackUrl = useMemo(() => 
    product.fallback_url || createFallbackImageUrl(product.name),
    [product.fallback_url, product.name]
  );
  
  // Seller fallback image URL
  const sellerFallbackUrl = useMemo(() => 
    product.seller_name ? `https://placehold.co/32x32/fe90ea/ffffff?text=${encodeURIComponent(product.seller_name.substring(0, 1).toUpperCase())}` : null,
    [product.seller_name]
  );

  // Check for mobile viewport size on mount and resize
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      if (mobile !== isMobile) {
        setIsMobile(mobile);
      }
    };
    
    // Debounce resize handler to improve performance
    const debouncedResize = () => {
      if (resizeHandlerRef.current) {
        clearTimeout(resizeHandlerRef.current);
      }
      resizeHandlerRef.current = setTimeout(checkMobile, 100);
    };
    
    // Set initial value
    checkMobile();
    
    // Add resize listener
    window.addEventListener('resize', debouncedResize);
    
    // Clean up
    return () => {
      window.removeEventListener('resize', debouncedResize);
      if (resizeHandlerRef.current) {
        clearTimeout(resizeHandlerRef.current);
      }
    };
  }, [isMobile]);
  
  // Add this useEffect for robust image loading with fallbacks
  useEffect(() => {
    // Set up image loading with multiple fallbacks
    const tryLoadImage = (url, fallbackIndex = 0) => {
      const img = new Image();
    
      img.onload = () => {
        setImageUrl(url);
        setImageLoaded(true);
        setImageError(false);
      };
      
      img.onerror = () => {
        const bgColors = ['212121', '4a4a4a', '6b6b6b', '444', '333', '555', 'abd123', 'fe90ea', '256789', '742d1e'];
        const textColors = ['ffffff', 'f0f0f0', 'eeeeee', 'dddddd', 'cccccc'];
        
        // Select random colors from our arrays
        const bgColor = bgColors[Math.floor(Math.random() * bgColors.length)];
        const textColor = textColors[Math.floor(Math.random() * textColors.length)];

        console.log(`Image failed to load: ${url}`);
        // Try fallbacks in order
        const fallbacks = [
          product.thumbnail_url, // Original URL
          createFallbackImageUrl(product.name), // Placeholder with name
          `https://placehold.co/600x400/${bgColor}/${textColor}?text=${encodeURIComponent(product.name.substring(0, 20))}`
        ];
        
        if (fallbackIndex < fallbacks.length - 1) {
          console.log(`Trying fallback ${fallbackIndex + 1}: ${fallbacks[fallbackIndex + 1]}`);
          tryLoadImage(fallbacks[fallbackIndex + 1], fallbackIndex + 1);
        } else {
          // All fallbacks failed
          console.log('All image fallbacks failed');
          setImageError(true);
          setImageLoaded(true); // Set as loaded so UI isn't stuck
        }
      };
      
      img.src = url;
    };
    
    // Reset loading state when we start
    setImageLoaded(false);
    setImageError(false);
    
    // Start with our current imageUrl
    tryLoadImage(imageUrl);
  }, [product.thumbnail_url, product.name]);
  
  // Similar functionality for seller thumbnail
  useEffect(() => {
    if (!product.seller_thumbnail) {
      setSellerImageLoaded(false);
      setSellerImageError(true);
      return;
    }
    
    const img = new Image();
    
    img.onload = () => {
      setSellerThumbnailUrl(product.seller_thumbnail);
      setSellerImageLoaded(true);
      setSellerImageError(false);
    };
    
    img.onerror = () => {
      console.log(`Seller image failed to load: ${product.seller_thumbnail}`);
      setSellerImageError(true);
      setSellerImageLoaded(true); // Set as loaded so UI isn't stuck
    };
    
    img.src = product.seller_thumbnail;
  }, [product.seller_thumbnail]);
  
  const handleMouseEnter = (e) => {
    // Don't trigger hover behavior on mobile
    if (isMobile) return;
    
    // Call the parent's hover handler
    if (onHover) onHover(product, e);
    
    // Set our local hover state
    setIsHovered(true);
  };
    
  const handleMouseLeave = () => {
    // Don't trigger hover behavior on mobile
    if (isMobile) return;
    
    // Call the parent's leave handler
    if (onLeave) onLeave();
    
    // Reset our local hover state
    setIsHovered(false);
    setMouseOverImage(false);
  };
    
  // Check if product contains design-related keywords
  const isDesignRelated = useMemo(() => {
    const keywords = ['design', 'image', 'logo', 'picture', 'stitch'];
    const searchText = `${product.name} ${product.description || ''}`.toLowerCase();
    return keywords.some(keyword => searchText.includes(keyword));
  }, [product.name, product.description]);
  
  // Determine if the default display should use hover mode
  // (only applies for design-related narrow images AND not for mobile)
  const useDefaultHoverMode = !isMobile && isDesignRelated && !isWideImage;
  
  // Determine if the card should show hover effects right now
  // (either actively hovered or using default hover mode)
  const shouldShowAsHover = !isMobile && (isHovered || useDefaultHoverMode);
  
  // Effect to calculate and store card height on mount
  useEffect(() => {
    if (cardRef.current) {
      // Use a fixed height for cards that better matches the reference images
      // Use a slightly smaller height on mobile
      const height = isMobile ? 320 : 300;
      setCardHeight(height);
    }
  }, [isMobile]);
  
  // Handle image loading and errors
  const handleImageLoad = (e) => {
    // Capture intrinsic dimensions
    const { naturalWidth, naturalHeight } = e.target;
    setImageDimensions({
      width: naturalWidth,
      height: naturalHeight
    });
    
    // Check if aspect ratio is wider than 4/3 (1.33)
    const aspectRatio = naturalWidth / naturalHeight;
    
    // Consider any image with aspect ratio > 4/3 as "wide"
    const isWide = aspectRatio > 1.33;
    setIsWideImage(isWide);
    
    // For wide images, we'll show the standard description even during hover
    setShowStandardOnHover(isWide);
    
    // Mark as loaded
    setImageLoaded(true);
  };
  
  const handleImageError = () => {
    console.log(`Image error for ${product.name} with URL: ${imageUrl}`);
    // We'll now use the fallbackUrl directly
    setImageError(true);
    
    // Try to load directly from the original URL as a last resort
    if (product.thumbnail_url && imageUrl !== product.thumbnail_url) {
      const img = new Image();
      img.onload = () => {
        // If the original URL loads successfully, use it
        setImageUrl(product.thumbnail_url);
        setImageError(false);
      };
      img.src = product.thumbnail_url;
    }
  };
  
  // Effect to handle delayed state changes
  useEffect(() => {
    // Clear any existing timeouts
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    if (shouldShowAsHover) {
      // Immediately hide details when hovering starts
      setShowDetails(false);
    } else {
      // When hover ends, wait for transition to complete before showing details
      timeoutRef.current = setTimeout(() => {
        setShowDetails(true);
      }, 50); // Small delay to ensure transitions complete
    }
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [shouldShowAsHover]);
  
  const handleImageMouseEnter = () => {
    if (isMobile) return;
    setMouseOverImage(true);
  };
  
  const handleImageMouseLeave = () => {
    if (isMobile) return;
    setMouseOverImage(false);
  };
  
  const handleImageClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setShowFullImage(true);
  };
  
  const handleCloseFullImage = () => {
    setShowFullImage(false);
  };
  
  // Seller component to replace score
  const SellerInfo = () => {
    // Only show if we have seller name or id
    if (!product.seller_name && !product.seller_id) return null;
    
    return (
      <div className="flex items-center">
        {/* Seller Avatar */}
        <div 
          className={`w-5 h-5 rounded-full overflow-hidden flex items-center justify-center mr-1 ${
            darkMode ? 'bg-gray-600' : 'bg-gray-200'
          }`}
        >
          {sellerImageError || !product.seller_thumbnail ? (
            <div 
              className="w-full h-full flex items-center justify-center text-xs font-bold"
              style={{ backgroundColor: '#FE90EA', color: 'white' }}
            >
              {product.seller_name ? product.seller_name.charAt(0).toUpperCase() : 'S'}
            </div>
          ) : (
            <img 
              src={sellerThumbnailUrl} 
              alt={product.seller_name || "Seller"}
              className="w-full h-full object-cover"
              onError={() => setSellerImageError(true)}
            />
          )}
        </div>
        
        {/* Seller Name */}
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
        ref={cardRef}
        className={`${darkMode ? 'bg-gray-700 border-gray-600 hover:border-[#FE90EA]' : 'bg-white border-gray-200 hover:border-[#FE90EA]'} border-2 rounded-lg overflow-hidden hover:shadow-lg transition-shadow product-card`}
        style={{ 
          height: cardHeight ? `${cardHeight}px` : 'auto', 
          position: 'relative',
          maxHeight: isMobile ? '300px' : '300px'
        }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={(e) => {
            if (isMobile && isDesignRelated) {
              handleImageClick(e);
            } else if (!(mouseOverImage && shouldShowAsHover) && !e.defaultPrevented) {
              window.open(product.url || "#", "_blank");
            }
          }}>
          <div 
            style={{ 
              position: shouldShowAsHover ? 'absolute' : 'relative',
              top: 0,
              left: 0,
              width: '100%',
              height: shouldShowAsHover ? '100%' : '160px',
              padding: shouldShowAsHover ? '8px' : '0',
              transition: 'padding 0.3s ease, height 0.3s ease',
              zIndex: 1,
              display: 'flex',
              alignItems: shouldShowAsHover ? 'flex-start' : 'center',
              justifyContent: 'center',
              cursor: shouldShowAsHover && mouseOverImage ? 'zoom-in' : 'pointer',
              backgroundColor: darkMode ? '#2D3748' : '#F7FAFC',  // Background while loading
            }}
            onMouseEnter={handleImageMouseEnter}
            onMouseLeave={handleImageMouseLeave}
            onClick={shouldShowAsHover && mouseOverImage ? handleImageClick : undefined}
          >
            {/* Loading indicator */}
            {!imageLoaded && !imageError && (
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                opacity: 0.7
              }}>
                <div className={`${darkMode ? 'text-gray-200' : 'text-gray-600'} text-xs font-medium`}>Loading...</div>
              </div>
            )}
            
            {/* The actual image */}
            <img
              loading="lazy"
              fetchPriority="high"
              ref={imageRef}
              src={imageError ? fallbackUrl : imageUrl}
              alt={product.name}
              onLoad={handleImageLoad}
              onError={handleImageError}
              style={{
                width: '100%',
                height: '100%',
                objectFit: shouldShowAsHover ? 'contain' : 'cover',
                objectPosition: shouldShowAsHover ? 'top' : 'center',
                transition: 'object-fit 0.3s ease, object-position 0.3s ease',
                opacity: imageLoaded ? 1 : 0,  // Only show when loaded
                visibility: imageLoaded ? 'visible' : 'hidden',
              }}
            />
            
            {/* Magnifier icon - only visible when mouse is over the image during hover */}
            {shouldShowAsHover && mouseOverImage && (
              <div 
                style={{
                  position: 'absolute',
                  top: '16px',
                  right: '16px',
                  backgroundColor: darkMode ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 255, 255, 0.5)',
                  borderRadius: '50%',
                  width: '32px',
                  height: '32px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backdropFilter: 'blur(2px)',
                  boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
                }}
              >
                <svg 
                  width="20"
                  height="20"
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke={darkMode ? "white" : "black"} 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                >
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                  <line x1="11" y1="8" x2="11" y2="14"></line>
                  <line x1="8" y1="11" x2="14" y2="11"></line>
                </svg>
              </div>
            )}
          </div>
          
          {/* Overlay description - only visible when in hover mode AND not showing standard description on hover */}
          {shouldShowAsHover && !showStandardOnHover && (
            <div
              style={{
                position: 'absolute',
                bottom: '8px',
                left: '8px',
                right: '8px',
                zIndex: 20,
                backgroundColor: darkMode ? 'rgba(26, 32, 44, 0.85)' : 'rgba(255, 255, 255, 0.85)',
                padding: '8px',
                borderRadius: '6px',
                backdropFilter: 'blur(2px)',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
              }}
            >
              <div className="flex justify-between items-start mb-1">
                <h3 className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-800'} line-clamp-1 max-w-[70%]`}>
                  {product.name}
                </h3>
                
                {/* Score tag in hover mode */}
                {product.score !== undefined && (
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded-full bg-red-600 text-white text-xs font-medium">
                    {typeof product.score === 'number' ? product.score.toFixed(2) : product.score}
                  </span>
                )}
              </div>
              
              <div className="flex items-center mb-1">
                {product.ratings_score !== undefined && (
                  <>
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
                  </>
                )}
              </div>
              <div className="flex items-center justify-between mt-1">
                <SellerInfo />
                
                {product.price_cents !== undefined && (
                  <span className={`text-xs font-medium ${darkMode ? 'text-gray-200' : 'text-gray-700'} mr-2`}>
                    ${(product.price_cents / 100).toFixed(2)}
                  </span>
                )}
                
                <a 
                  href={product.url || "#"} 
                  target="#"
                  className="inline-flex items-center justify-center px-2 py-1 text-xs font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-1 focus:ring-[#FE90EA] border border-black"
                  onClick={(e) => e.stopPropagation()}
                >
                  View details
                </a>
              </div>
            </div>
          )}
          
          {/* Price tag - always visible when not in hover mode OR when showing standard description on hover */}
          {product.price_cents !== undefined && (!shouldShowAsHover || showStandardOnHover) && (
            <div className="absolute rounded-md top-3 right-3 flex items-center" style={{ zIndex: 30 }}>
              <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-base border border-t-transparent border-l-black border-r-transparent border-b-black">
                ${(product.price_cents / 100).toFixed(2)}
                <div className="absolute -right-[4px] -top-[1px] w-0 h-0 border-t-[8px] border-b-[7px] border-l-[5px] border-t-transparent border-b-transparent border-l-black"></div>
                <div className="absolute -right-[4px] bottom-[1px] w-0 h-0 border-t-[7px] border-b-[7px] border-l-[5px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
              </div>
            </div>
          )}
          
          {/* Design related label - only visible for design products when not hovered */}
          {isDesignRelated && !isHovered && (
            <div 
              className={`absolute top-2 left-2 text-xs font-medium px-1.5 py-0.5 rounded-full ${
                darkMode ? 'bg-blue-600 text-white' : 'bg-blue-100 text-blue-800'
              }`}
              style={{ zIndex: 25 }}
            >
              Design
            </div>
          )}
          
          {/* Score tag - always visible when not in hover mode */}
          {product.score !== undefined && !shouldShowAsHover && (
            <div 
              className="absolute top-2 left-2 text-xs font-medium px-1.5 py-0.5 rounded-full bg-red-600 text-white"
              style={{ 
                zIndex: 25,
                left: isDesignRelated ? '60px' : '10px' // Position to the right of Design tag if present
              }}
            >
              Score: {typeof product.score === 'number' ? product.score.toFixed(2) : product.score}
            </div>
          )}
          
          {/* Standard details section - visible when not in hover mode OR when showing standard description on hover */}
          {(showDetails && !shouldShowAsHover) || (showStandardOnHover && isHovered) ? (
            <div 
              style={{ 
                padding: '0.75rem',
                opacity: isHovered && showStandardOnHover ? 0.9 : 1,
                backgroundColor: isHovered && showStandardOnHover ? (darkMode ? 'rgba(26, 32, 44, 0.9)' : 'rgba(255, 255, 255, 0.9)') : 'transparent',
                position: isHovered && showStandardOnHover ? 'absolute' : 'relative',
                bottom: 0,
                left: 0,
                right: 0,
                zIndex: isHovered && showStandardOnHover ? 20 : 1,
                backdropFilter: isHovered && showStandardOnHover ? 'blur(2px)' : 'none',
              }}
            >
              <h3 className={`font-medium text-sm ${darkMode ? 'text-gray-100' : 'text-gray-800'} mb-1 line-clamp-1`}>{product.name}</h3>
              
              {/* Rating display with stars */}
              { product.ratings_count > 0 && product.ratings_score !== undefined && (
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
                {/* Replace score with seller info */}
                <SellerInfo />
                
                <a 
                  href={product.url || "#"} 
                  target="#"
                  className="inline-flex items-center justify-center px-2 py-1 text-xs font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-1 focus:ring-[#FE90EA] border border-black"
                >
                  View details
                </a>
              </div>
            </div>
          ) : null}
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
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
            
            <img 
              src={imageError ? fallbackUrl : imageUrl}
              alt={product.name}
              className="max-w-full max-h-full object-contain cursor-pointer"
              onClick={() => window.open(product.url || "#", "_blank")}
              onError={handleImageError}
            />
                  
            <div className="absolute bottom-4 left-0 right-0 text-center text-white bg-black bg-opacity-50 py-2 px-4">
              <h3 className="font-bold text-base">{product.name}</h3>
              <div className="flex items-center justify-center mt-1">
                {product.seller_name && (
                  <div className="flex items-center mr-3">
                    <div className="w-5 h-5 rounded-full overflow-hidden mr-1 bg-[#FE90EA] flex items-center justify-center">
                      {sellerImageError || !product.seller_thumbnail ? (
                        <span className="text-xs font-bold text-white">
                          {product.seller_name.charAt(0).toUpperCase()}
                        </span>
                      ) : (
                        <img 
                          src={sellerThumbnailUrl} 
                          alt={product.seller_name} 
                          className="w-full h-full object-cover"
                        />
                      )}
                    </div>
                    <span className="text-sm opacity-90">{product.seller_name}</span>
                  </div>
                )}
                <p className="text-sm opacity-90">
                  {product.price_cents ? `$${(product.price_cents / 100).toFixed(2)}` : 'Free'}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default ProductCard;