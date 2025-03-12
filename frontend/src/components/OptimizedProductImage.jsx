import React, { useState, useEffect } from 'react';
import { createFallbackImageUrl } from '../services/imageService';

const OptimizedProductImage = ({ 
  src, 
  alt, 
  index = 0,
  fallbackSrc,
  onLoad, 
  onError,
  className = '',
  style = {},
  width = 300,
  height = 200,
  objectFit = 'cover'
}) => {
  const [error, setError] = useState(false);
  const [loaded, setLoaded] = useState(false);
  
  // Determine loading priority based on index
  const loadingStrategy = index < 6 ? "eager" : "lazy";
  const fetchPriority = index < 3 ? "high" : "auto";
  
  // Choose final src based on error state
  const finalSrc = error ? (fallbackSrc || createFallbackImageUrl(alt)) : src;
  
  const handleImageError = (e) => {
    setError(true);
    if (onError) onError(e);
  };

  const handleImageLoad = (e) => {
    setLoaded(true);
    if (onLoad) onLoad(e);
  };
  
  // Try to generate WebP version if it's a placeholder URL
  const webpSrc = finalSrc?.includes('placehold.co') ? 
    finalSrc.replace('placehold.co', 'placehold.co') + '&format=webp' : 
    finalSrc;
    
  return (
    <div className={`image-container ${className}`} style={{ position: 'relative', overflow: 'hidden', ...style }}>
      {!loaded && (
        <div className="placeholder-loader" style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#f0f0f0',
          color: '#666',
          fontSize: '0.75rem'
        }}>
          <div className="loading-indicator"></div>
        </div>
      )}
      
      <picture>
        {/* Provide WebP if available */}
        {webpSrc && <source srcSet={webpSrc} type="image/webp" />}
        
        {/* Traditional image as fallback */}
        <img
          src={finalSrc}
          alt={alt || 'Product image'}
          loading={loadingStrategy}
          fetchpriority={fetchPriority}
          onLoad={handleImageLoad}
          onError={handleImageError}
          style={{
            width: '100%',
            height: '100%',
            objectFit: objectFit,
            opacity: loaded ? 1 : 0,
            transition: 'opacity 0.2s ease-in-out'
          }}
          width={width}
          height={height}
        />
      </picture>
    </div>
  );
};

export default OptimizedProductImage;