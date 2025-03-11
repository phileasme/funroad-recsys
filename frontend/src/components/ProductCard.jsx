import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createFallbackImageUrl, getProxiedImageUrl } from '../services/imageService';

const ProductCard = React.memo(({ product, index, darkMode, onHover, onLeave }) => {
  // State to track which view mode is active
  const [showImageView, setShowImageView] = useState(false);
  // State to track the initial (default) view mode
  const [initialViewIsImage, setInitialViewIsImage] = useState(false);
  // State to track if card is hovered
  const [isHovered, setIsHovered] = useState(false);
  // State to control fullscreen image view
  const [showFullImage, setShowFullImage] = useState(false);
  // State to track if mouse is over image specifically
  const [mouseOverImage, setMouseOverImage] = useState(false);
  // State to track image aspect ratio
  const [isWideImage, setIsWideImage] = useState(false);
  const [isTallImage, setIsTallImage] = useState(false);
  const [isSquareImage, setIsSquareImage] = useState(false);
  // State to track window width for responsive behavior
  const [isMobile, setIsMobile] = useState(false);
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
  const hoverTimerRef = useRef(null);
  const imageRef = useRef(null);
  const resizeHandlerRef = useRef(null);
  
  // Store the original height of the card
  const [cardHeight, setCardHeight] = useState(null);

  // Check if product contains design-related or art-related keywords
  const isArtRelated = useMemo(() => {
    const keywords = ['design', 'image', 'logo', 'picture', 'stitch', 'photography', 'photo', 'mockup', 'template'];
    const searchText = `${product.name} ${product.description || ''}`.toLowerCase();
    return keywords.some(keyword => searchText.includes(keyword));
  }, [product.name, product.description]);
  
  // Function to safely handle URIs
  const sanitizeForURI = (str) => {
    if (!str) return '';
    return str.replace(/[\u0000-\u001F\u007F-\u009F\u2000-\u200F]/g, '')
             .replace(/[^\w\s-.,]/g, ' ')
             .trim();
  };
  
  // Generate fallback URLs
  const fallbackUrl = useMemo(() => 
    product.fallback_url || createFallbackImageUrl(sanitizeForURI(product.name)),
    [product.fallback_url, product.name]
  );
  
  const sellerFallbackUrl = useMemo(() => 
    product.seller_name 
      ? `https://placehold.co/32x32/fe90ea/ffffff?text=${encodeURIComponent(sanitizeForURI(product.seller_name).substring(0, 1).toUpperCase())}`
      : null,
    [product.seller_name]
  );

  // Function to toggle view mode manually
  const toggleView = () => {
    // Only allow toggling if not on mobile and not a wide image
    if (!isMobile && !(isArtRelated && isWideImage)) {
      setShowImageView(prev => !prev);
    }
  };

  // Initialize card view based on art detection and image type
  useEffect(() => {
    // Set initial state based on image dimensions
    if (!isMobile) {
      if (isTallImage || isSquareImage) {
        // For art-related tall or square images, default to details view
        setShowImageView(true);
        setInitialViewIsImage(true);
      } else {
        setShowImageView(false);
        setInitialViewIsImage(false);
      }
    } else {
      // For non-art content, always default to details view
      setShowImageView(false);
      setInitialViewIsImage(false);
    }
  }, [isArtRelated, isMobile, isTallImage, isSquareImage, isWideImage]);
  

  // Check for mobile viewport size on mount and resize
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      if (mobile !== isMobile) {
        setIsMobile(mobile);
      }
    };
    
    const debouncedResize = () => {
      if (resizeHandlerRef.current) {
        clearTimeout(resizeHandlerRef.current);
      }
      resizeHandlerRef.current = setTimeout(checkMobile, 100);
    };
    
    checkMobile();
    window.addEventListener('resize', debouncedResize);
    
    return () => {
      window.removeEventListener('resize', debouncedResize);
      if (resizeHandlerRef.current) {
        clearTimeout(resizeHandlerRef.current);
      }
    };
  }, [isMobile]);
  
  // Image loading with fallbacks
  useEffect(() => {
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
        
        const bgColor = bgColors[Math.floor(Math.random() * bgColors.length)];
        const textColor = textColors[Math.floor(Math.random() * textColors.length)];

        console.log(`Image failed to load: ${url}`);
        const fallbacks = [
          product.thumbnail_url,
          createFallbackImageUrl(product.name),
          `https://placehold.co/600x400/${bgColor}/${textColor}?text=${encodeURIComponent(product.name.substring(0, 20))}`
        ];
        
        if (fallbackIndex < fallbacks.length - 1) {
          console.log(`Trying fallback ${fallbackIndex + 1}: ${fallbacks[fallbackIndex + 1]}`);
          tryLoadImage(fallbacks[fallbackIndex + 1], fallbackIndex + 1);
        } else {
          console.log('All image fallbacks failed');
          setImageError(true);
          setImageLoaded(true);
        }
      };
      
      img.src = url;
    };
    
    setImageLoaded(false);
    setImageError(false);
    tryLoadImage(imageUrl);
  }, [product.thumbnail_url, product.name]);
  
  // Seller thumbnail loading
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
      setSellerImageLoaded(true);
    };
    
    img.src = product.seller_thumbnail;
  }, [product.seller_thumbnail]);
  
  // Mouse enter handler - with view toggle on hover
  const handleMouseEnter = (e) => {
    if (isMobile) return;
    
    console.log(`ProductCard mouseEnter: ${product.id || product.name}`);
    setIsHovered(true);
    
    // Only toggle view for art-related content that's tall or square
    if (isTallImage || isSquareImage) {
      // Clear any existing hover timers
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
      }
      
      // Use a short delay to prevent rapid toggling
      hoverTimerRef.current = setTimeout(() => {
        // Toggle to opposite of initial state on hover
        setShowImageView(!initialViewIsImage);
      }, 50);
    }
    
    // Don't toggle for wide images (more width than height)
    // This keeps them in their default state
    
    // Call the parent's hover handler if provided
    if (onHover) onHover(product, e);
  };
    
  // Mouse leave handler
  const handleMouseLeave = () => {
    if (isMobile) return;
    
    setIsHovered(false);
    setMouseOverImage(false);
    
    // Clear any existing hover timers
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
    }
    
    // Reset to initial view state on mouse leave, but only for togglable cards
    if (isTallImage || isSquareImage) {
      setShowImageView(initialViewIsImage);
    }
    
    // Call the parent's leave handler if provided
    if (onLeave) onLeave();
  };
  
    
  // Calculate card height
  useEffect(() => {
    if (cardRef.current) {
      const height = isMobile ? 320 : 300;
      setCardHeight(height);
    }
  }, [isMobile]);
  
  // Handle image loading and errors
  const handleImageLoad = (e) => {
    const { naturalWidth, naturalHeight } = e.target;
    setImageDimensions({
      width: naturalWidth,
      height: naturalHeight
    });
    
    // Calculate aspect ratio
    const aspectRatio = naturalWidth / naturalHeight;
    console.log(`Image for ${product.name} has aspect ratio: ${aspectRatio.toFixed(2)}`);
    
    // Check image shape
    const isTall = aspectRatio < 0.9;
    setIsTallImage(isTall);
    
    const isSquare = aspectRatio >= 0.9 && aspectRatio <= 1.1;
    setIsSquareImage(isSquare);
    
    const isWide = aspectRatio > 1.33;
    setIsWideImage(isWide);
    
    // Mark as loaded
    setImageLoaded(true);
    
    // Update initial view mode based on image type (if art-related)
    if (isArtRelated && !isMobile) {
      // For tall/square images, we prefer details view initially
      if (isTall || isSquare) {
        setInitialViewIsImage(false);
        setShowImageView(false);
      } else {
        // For wide images, we prefer image view initially
        setInitialViewIsImage(true);
        setShowImageView(true);
      }
    }
  };
  
  const handleImageError = () => {
    console.log(`Image error for ${product.name} with URL: ${imageUrl}`);
    setImageError(true);
    
    if (product.thumbnail_url && imageUrl !== product.thumbnail_url) {
      const img = new Image();
      img.onload = () => {
        setImageUrl(product.thumbnail_url);
        setImageError(false);
      };
      img.src = product.thumbnail_url;
    }
  };
  
  // Image-specific mouse events
  const handleImageMouseEnter = () => {
    if (isMobile) return;
    setMouseOverImage(true);
  };
  
  const handleImageMouseLeave = () => {
    if (isMobile) return;
    setMouseOverImage(false);
  };
  
  // Fullscreen image handling
  const handleImageClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setShowFullImage(true);
  };
  
  const handleCloseFullImage = () => {
    setShowFullImage(false);
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
      }
    };
  }, []);
  
  // Seller information component
  const SellerInfo = () => {
    if (!product.seller_name && !product.seller_id) return null;
    
    return (
      <div className="flex items-center">
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
        
        {product.seller_name && (
          <span className={`text-xs ${darkMode ? 'text-gray-300' : 'text-gray-600'} truncate max-w-[80px]`}>
            {product.seller_name}
          </span>
        )}
      </div>
    );
  };
  
  // Helper to display score in a consistent format
  const getDisplayScore = () => {
    if (product.displayScore !== undefined) {
      const label = product.scoreLabel ? `${product.scoreLabel}: ` : '';
      return `${label}${product.displayScore}`;
    }
    
    if (product.score !== undefined) {
      const scoreValue = typeof product.score === 'number' ? 
        product.score.toFixed(4) : product.score;
      return `Score: ${scoreValue}`;
    }
    
    return "N/A";
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
            if (isMobile) {
              // For mobile, clicking the card will now just navigate to product URL
              // The image section handles the magnify click functionality
              if (!e.defaultPrevented) {
                window.open(product.url || "#", "_blank");
              }
            } else if (!(mouseOverImage && showImageView) && !e.defaultPrevented) {
              window.open(product.url || "#", "_blank");
            }
          }}
      >
        {/* Image section */}
        <div 
          style={{ 
            position: showImageView ? 'absolute' : 'relative',
            top: 0,
            left: 0,
            width: '100%',
            height: showImageView ? '100%' : '160px',
            padding: showImageView ? '8px' : '0',
            transition: 'padding 0.3s ease, height 0.3s ease',
            zIndex: 1,
            display: 'flex',
            alignItems: showImageView ? 'flex-start' : 'center',
            justifyContent: 'center',
            cursor: 'zoom-in', // Changed to always show zoom-in cursor
            backgroundColor: darkMode ? '#2D3748' : '#F7FAFC',
          }}
          onMouseEnter={handleImageMouseEnter}
          onMouseLeave={handleImageMouseLeave}
          onClick={handleImageClick} // Always open full image on click
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
          
          {/* The image */}
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
              objectFit: showImageView ? 'contain' : 'cover',
              objectPosition: showImageView ? 'top' : 'center',
              transition: 'object-fit 0.3s ease, object-position 0.3s ease',
              opacity: imageLoaded ? 1 : 0,
              visibility: imageLoaded ? 'visible' : 'hidden',
            }}
          />
          
          {/* Magnifier icon - ALWAYS visible now */}
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
          
          {/* Toggle view button */}
          {!isMobile && (isArtRelated || isTallImage || isSquareImage) &&  (
            <div 
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
            </div>
          )}
        </div>
        
        {/* Overlay description - only in image view mode */}
        {showImageView && (
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
        
        {/* Price tag - visible in standard mode */}
        {product.price_cents !== undefined && (
          <div className="absolute rounded-md top-3 right-3 flex items-center" style={{ zIndex: 30 }}>
            <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-base border border-t-transparent border-l-black border-r-transparent border-b-black">
              ${(product.price_cents / 100).toFixed(2)}
              <div className="absolute -right-[4px] -top-[1px] w-0 h-0 border-t-[8px] border-b-[7px] border-l-[5px] border-t-transparent border-b-transparent border-l-black"></div>
              <div className="absolute -right-[4px] bottom-[1px] w-0 h-0 border-t-[7px] border-b-[7px] border-l-[5px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
            </div>
          </div>
        )}
        
        {/* Standard details section - visible in standard mode */}
        {!showImageView && (
          <div 
            style={{ 
              padding: '0.75rem',
              position: 'relative',
              bottom: 0,
              left: 0,
              right: 0,
              zIndex: 1,
            }}
          >
            <h3 className={`font-medium text-sm ${darkMode ? 'text-gray-100' : 'text-gray-800'} mb-1 line-clamp-1`}>
              {product.name}
            </h3>
            
            {/* Rating display with stars */}
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
                target="#"
                className="inline-flex items-center justify-center px-2 py-1 text-xs font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-1 focus:ring-[#FE90EA] border border-black"
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
                  
            <div className="absolute bottom-4 left-0 right-0 text-center text-white bg-black bg-opacity-30 py-2 px-4">
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
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
});

export default ProductCard;