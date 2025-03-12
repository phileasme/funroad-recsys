// src/services/enhancedImageQueue.js

// Constants for configuration
const CONFIG = {
    MAX_CONCURRENT_LOADS: 4,      // Maximum number of images loading simultaneously
    LOAD_BATCH_SIZE: 6,           // Process this many images at once when checking queue
    PRELOAD_VISIBLE_VIEWPORT: 2,  // Preload images within X times the viewport height
    THROTTLE_INTERVAL: 150,       // Milliseconds between queue processing batches
    USE_FETCH_PRIORITY: true,     // Whether to use fetchpriority attribute
    OBSERVER_THRESHOLD: [0, 0.1, 0.5, 1.0], // Multiple thresholds for more granular loading
  };
  
  // Image queue system with priority levels
  const imageQueues = {
    critical: [],    // Top fold, currently visible
    high: [],        // Just below the fold, about to be visible
    normal: [],      // Standard priority for images that will be needed soon
    low: []          // Low priority background loading
  };
  
  // Track loading state
  let currentlyLoading = 0;
  let observerInitialized = false;
  const observedElements = new Set();
  const loadedImages = new Set();
  const processTimer = { current: null };
  
  // Support detection
  const supportsFetchPriority = 'fetchpriority' in HTMLImageElement.prototype;
  const supportsModernFormats = {
    webp: false,
    avif: false
  };
  
  // Detect format support on init
  function detectFormatSupport() {
    // WebP detection
    const webpImage = new Image();
    webpImage.onload = () => { supportsModernFormats.webp = true; };
    webpImage.src = 'data:image/webp;base64,UklGRh4AAABXRUJQVlA4TBEAAAAvAAAAAAfQ//73v/+BiOh/AAA=';
    
    // AVIF detection
    const avifImage = new Image();
    avifImage.onload = () => { supportsModernFormats.avif = true; };
    avifImage.src = 'data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAAB0AAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAIAAAACAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgQ0MAAAAABNjb2xybmNseAACAAIAAYAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAACVtZGF0EgAKCBgANogQEAwgMg8f8D///8WfhwB8+ErK42A=';
  }
  
  // Check if this image should use WebP/AVIF
  function getOptimizedImageUrl(originalUrl) {
    // Skip for data URLs, SVGs, or if format detection hasn't completed
    if (originalUrl.startsWith('data:') || 
        originalUrl.endsWith('.svg') || 
        (!supportsModernFormats.webp && !supportsModernFormats.avif)) {
      return originalUrl;
    }
    
    // Replace with WebP or AVIF if supported (you'll need server support for this)
    // This assumes your server can deliver the appropriate format when requested
    // Example: https://example.com/image.jpg?format=webp
    
    if (supportsModernFormats.avif) {
      return appendFormatParam(originalUrl, 'avif');
    } 
    else if (supportsModernFormats.webp) {
      return appendFormatParam(originalUrl, 'webp');
    }
    
    return originalUrl;
  }
  
  // Helper to append format parameter to URLs
  function appendFormatParam(url, format) {
    // Don't modify URLs that already specify a format
    if (url.includes('format=')) return url;
    
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}format=${format}`;
  }
  
  // Add image to appropriate queue based on priority
  export function queueImageLoad(imgElement, priority = 'normal') {
    if (!imgElement || !imgElement.dataset || !imgElement.dataset.src) return;
    
    // Skip if this image is already loaded or queued
    if (loadedImages.has(imgElement.dataset.src) || imageQueues[priority].includes(imgElement)) {
      return;
    }
    
    // Add to appropriate queue
    if (priority in imageQueues) {
      imageQueues[priority].push(imgElement);
    } else {
      imageQueues.normal.push(imgElement);
    }
    
    // Ensure queue is being processed
    ensureQueueProcessing();
  }
  
  // Start/ensure queue processing
  function ensureQueueProcessing() {
    if (processTimer.current) return;
    
    processQueue();
  }
  
  // Process queues with priority order
  function processQueue() {
    // Clear any existing timer
    if (processTimer.current) {
      clearTimeout(processTimer.current);
      processTimer.current = null;
    }
    
    // If we're at capacity, schedule next check and return
    if (currentlyLoading >= CONFIG.MAX_CONCURRENT_LOADS) {
      processTimer.current = setTimeout(processQueue, CONFIG.THROTTLE_INTERVAL);
      return;
    }
    
    // Count how many more images we can load
    const slotsAvailable = CONFIG.MAX_CONCURRENT_LOADS - currentlyLoading;
    if (slotsAvailable <= 0) return;
    
    // Process queues in priority order
    let processed = 0;
    const queuePriorities = ['critical', 'high', 'normal', 'low'];
    
    for (const priority of queuePriorities) {
      const queue = imageQueues[priority];
      
      while (queue.length > 0 && processed < slotsAvailable) {
        const img = queue.shift();
        
        // Skip if element is not valid or not in DOM anymore
        if (!isValidImageElement(img)) continue;
        
        // Start loading this image
        loadImage(img, priority);
        processed++;
        
        // If we've reached batch size, break out
        if (processed >= Math.min(slotsAvailable, CONFIG.LOAD_BATCH_SIZE)) break;
      }
      
      // If we've loaded enough for this batch, stop
      if (processed >= Math.min(slotsAvailable, CONFIG.LOAD_BATCH_SIZE)) break;
    }
    
    // Schedule next run if there are more images to process
    const hasMoreImages = Object.values(imageQueues).some(q => q.length > 0);
    
    if (hasMoreImages || currentlyLoading > 0) {
      processTimer.current = setTimeout(processQueue, CONFIG.THROTTLE_INTERVAL);
    } else {
      processTimer.current = null;
    }
  }
  
  // Check if image element is valid and still needs loading
  function isValidImageElement(img) {
    return img && 
           img.dataset && 
           img.dataset.src && 
           !img.dataset.loaded && 
           document.body.contains(img);
  }
  
  // Load a single image
  function loadImage(img, priority) {
    if (!isValidImageElement(img)) return;
    
    const src = img.dataset.src;
    currentlyLoading++;
    
    // Set fetchpriority attribute if supported and high priority
    if (CONFIG.USE_FETCH_PRIORITY && supportsFetchPriority) {
      if (priority === 'critical' || priority === 'high') {
        img.fetchpriority = priority === 'critical' ? 'high' : 'auto';
      }
    }
    
    // If image has dimensions, set them to prevent layout shifts
    if (img.dataset.width && img.dataset.height) {
      img.width = img.dataset.width;
      img.height = img.dataset.height;
    }
    
    // Get potentially optimized URL (WebP/AVIF)
    const optimizedSrc = getOptimizedImageUrl(src);
    
    // Set onload/onerror handlers
    img.onload = () => {
      imageLoaded(img, src);
    };
    
    img.onerror = () => {
      // If we tried an optimized format and it failed, fallback to original
      if (optimizedSrc !== src) {
        console.log(`Optimized format failed, falling back to original: ${src}`);
        img.src = src;
      } else {
        // Original format also failed
        imageFailed(img, src);
      }
    };
    
    // Mark loading started
    img.dataset.loading = 'true';
    
    // Set src to start loading
    img.src = optimizedSrc;
    
    // Remove data-src to prevent re-queueing
    delete img.dataset.src;
  }
  
  // Handle successful image load
  function imageLoaded(img, src) {
    currentlyLoading--;
    
    // Mark as loaded
    img.dataset.loaded = 'true';
    delete img.dataset.loading;
    img.classList.add('image-loaded');
    
    // Add to loaded set to avoid reloading
    loadedImages.add(src);
    
    // Update performance metrics if desired
    img.dataset.loadTime = performance.now();
  }
  
  // Handle failed image load
  function imageFailed(img, src) {
    console.warn(`Failed to load image: ${src}`);
    currentlyLoading--;
    
    // Mark as failed but still remove from loading state
    img.dataset.loadFailed = 'true';
    delete img.dataset.loading;
    
    // Add to loaded set to avoid reloading failures repeatedly
    loadedImages.add(src);
  }
  
  // Initialize intersection observer with improved settings
  export function initializeObserver() {
    // Only initialize once
    if (observerInitialized) return;
    
    // Check browser support
    if (!('IntersectionObserver' in window)) {
      console.log('IntersectionObserver not supported, loading all images directly');
      return;
    }
    
    // Detect modern format support
    detectFormatSupport();
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const card = entry.target;
        const img = card.querySelector('img[data-src]');
        
        if (!img) return;
        
        // Calculate priority based on intersection ratio and position
        if (entry.isIntersecting) {
          const rect = card.getBoundingClientRect();
          const viewportHeight = window.innerHeight;
          
          // Determine priority based on visibility and position
          let priority;
          if (entry.intersectionRatio > 0.5 || 
              (rect.top >= 0 && rect.bottom <= viewportHeight)) {
            // Fully or mostly visible - critical priority
            priority = 'critical';
          } else if (rect.top < viewportHeight * CONFIG.PRELOAD_VISIBLE_VIEWPORT) {
            // Just below the fold - high priority
            priority = 'high';
          } else {
            // Further down - normal priority
            priority = 'normal';
          }
          
          // Queue this image with appropriate priority
          queueImageLoad(img, priority);
          
          // If it's critical or high priority, stop observing immediately
          if (priority === 'critical' || priority === 'high') {
            observer.unobserve(card);
            observedElements.delete(card);
          }
        }
      });
    }, { 
      // More granular thresholds for better prioritization
      rootMargin: '200px 0px 500px 0px', // Asymmetric - load more below viewport than above
      threshold: CONFIG.OBSERVER_THRESHOLD  // Multiple thresholds for better tracking
    });
    
    observerInitialized = true;
    
    // Expose observer for later use
    window.productImageObserver = observer;
    
    return observer;
  }
  
  // Observe a single product card element
  export function observeProductCard(cardElement) {
    if (!cardElement || observedElements.has(cardElement)) return;
    
    if (!observerInitialized) {
      initializeObserver();
    }
    
    if (window.productImageObserver) {
      window.productImageObserver.observe(cardElement);
      observedElements.add(cardElement);
    }
  }
  
  // Function to reset and re-observe all product cards with better prioritization
  export function refreshProductCardObserver() {
    // Find all product cards and observe them
    const productCards = document.querySelectorAll('.product-card');
    
    if (!observerInitialized) {
      initializeObserver();
    }
    
    if (window.productImageObserver) {
      // Clear previously observed elements
      observedElements.forEach(element => {
        window.productImageObserver.unobserve(element);
      });
      observedElements.clear();
      
      // Calculate viewport
      const viewportHeight = window.innerHeight;
      
      // Observe new elements with prioritization
      productCards.forEach((card, index) => {
        // Start observing
        window.productImageObserver.observe(card);
        observedElements.add(card);
        
        // Pre-queue the first few images directly
        if (index < 5) {
          const img = card.querySelector('img[data-src]');
          if (img) {
            const rect = card.getBoundingClientRect();
            // If in viewport, load immediately with high priority
            if (rect.top < viewportHeight) {
              queueImageLoad(img, 'critical');
            }
          }
        }
      });
      
      console.log(`Observing ${productCards.length} product cards for visibility-based loading`);
    }
  }
  
  // Preload a specific set of images (useful for critical path images)
  export function preloadImages(urls, highPriority = false) {
    if (!Array.isArray(urls) || urls.length === 0) return Promise.resolve([]);
    
    return Promise.all(urls.map((url, index) => {
      return new Promise((resolve) => {
        // Skip already loaded images
        if (loadedImages.has(url)) {
          resolve({ url, success: true, cached: true });
          return;
        }
        
        const img = new Image();
        
        // Set appropriate priority
        if (CONFIG.USE_FETCH_PRIORITY && supportsFetchPriority) {
          img.fetchpriority = highPriority || index < 3 ? 'high' : 'auto';
        }
        
        img.onload = () => {
          loadedImages.add(url);
          resolve({ url, success: true });
        };
        
        img.onerror = () => {
          resolve({ url, success: false });
        };
        
        // Set optimized URL if possible
        img.src = getOptimizedImageUrl(url);
      });
    }));
  }
  
  // Clear image cache (useful when refreshing data)
  export function clearImageCache() {
    loadedImages.clear();
  }
  
  // Initialize on import
  detectFormatSupport();
  
  // Make sure to export all functions
  export default {
    queueImageLoad,
    initializeObserver,
    observeProductCard,
    refreshProductCardObserver,
    preloadImages,
    clearImageCache
  };