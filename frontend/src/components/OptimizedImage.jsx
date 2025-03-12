import React, { useState, useEffect, useRef } from 'react';

/**
 * OptimizedImage - A simplified image component with lazy loading and error handling
 * 
 * @param {Object} props
 * @param {string} props.src - The primary image source URL
 * @param {string} props.alt - Alt text for the image
 * @param {string} props.fallbackSrc - Fallback image to use if primary fails
 * @param {string} props.className - CSS classes to apply to the image
 * @param {string} props.style - Inline styles to apply to the image
 * @param {string} props.objectFit - CSS object-fit property (cover, contain, etc.)
 * @param {boolean} props.priority - Whether this is a high priority image
 * @param {function} props.onLoad - Callback when image loads successfully
 * @param {function} props.onError - Callback when image fails to load
 */
const OptimizedImage = ({
  src,
  alt,
  fallbackSrc,
  className = '',
  style = {},
  objectFit = 'cover',
  priority = false,
  onLoad,
  onError,
  ...rest
}) => {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  const [currentSrc, setCurrentSrc] = useState(null);
  const imgRef = useRef(null);
  const observer = useRef(null);
  
  // Handle successful load
  const handleLoad = () => {
    setLoaded(true);
    if (onLoad && typeof onLoad === 'function') {
      onLoad();
    }
  };
  
  // Handle load error
  const handleError = () => {
    if (!error && fallbackSrc) {
      // Try fallback image if main source fails
      setError(true);
      setCurrentSrc(fallbackSrc);
    } else if (onError && typeof onError === 'function') {
      onError();
    }
  };

  // Set up intersection observer for lazy loading
  useEffect(() => {
    // If priority is true, don't use lazy loading
    if (priority) {
      setCurrentSrc(src);
      return;
    }
    
    // Only set up observer if IntersectionObserver is available
    if ('IntersectionObserver' in window) {
      observer.current = new IntersectionObserver(
        (entries) => {
          entries.forEach(entry => {
            if (entry.isIntersecting && !currentSrc) {
              setCurrentSrc(src);
              observer.current.disconnect();
            }
          });
        },
        {
          rootMargin: '200px 0px', // Load images before they enter viewport
          threshold: 0.01
        }
      );
      
      if (imgRef.current) {
        observer.current.observe(imgRef.current);
      }
    } else {
      // Fallback for browsers that don't support IntersectionObserver
      setCurrentSrc(src);
    }
    
    // Clean up observer on unmount
    return () => {
      if (observer.current) {
        observer.current.disconnect();
      }
    };
  }, [src, priority, currentSrc]);

  // Set up standard props for image
  const imageProps = {
    ref: imgRef,
    alt: alt || 'Image',
    className: `optimized-image ${loaded ? 'loaded' : 'loading'} ${className}`,
    style: {
      objectFit,
      opacity: loaded ? 1 : 0,
      transition: 'opacity 0.3s ease-in-out',
      ...style
    },
    onLoad: handleLoad,
    onError: handleError,
    // Use native loading attribute for most images
    loading: priority ? 'eager' : 'lazy',
    ...rest
  };

  // Only add src when we're ready to load
  if (currentSrc) {
    imageProps.src = currentSrc;
  }
  
  // Add fetchpriority if supported and requested
  if (priority && 'fetchpriority' in HTMLImageElement.prototype) {
    imageProps.fetchpriority = 'high';
  }

  return (
    <div className="image-container" style={{ position: 'relative', width: '100%', height: '100%' }}>
      {!loaded && !error && (
        <div className="image-placeholder" style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundColor: '#f0f0f0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          {/* Simple loading indicator */}
          <div className="loading-indicator" style={{
            width: '20px',
            height: '20px',
            border: '2px solid #ccc',
            borderTopColor: '#fe90ea',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
        </div>
      )}
      <img {...imageProps} alt={alt || 'Image'} />
      
      {/* Add animation keyframes */}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default OptimizedImage;