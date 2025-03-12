/**
 * Simplified Image Service - Focused on what matters most
 */

// Constants
const IMAGE_PROXY_PATH = '/image-proxy/';
const PLACEHOLDER_BASE_URL = 'https://placehold.co/600x400';
const BATCH_SIZE = 5; // Number of images to load in the first batch
const DELAY_BEFORE_SECOND_BATCH = 2000; // 2 seconds delay before loading the rest

/**
 * Generate a color from text for consistent placeholder colors
 * @param {string} text - Input text
 * @returns {string} - Hex color
 */
function getHashColor(text) {
  if (!text) return '#fe90ea'; // Default pink
  
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = text.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  // Generate a hue between 0 and 360
  const hue = Math.abs(hash) % 360;
  
  // Use HSL to ensure good saturation and lightness
  return `hsl(${hue}, 80%, 60%)`;
}

/**
 * Create a fallback image URL with encoded text
 * @param {string} text - Text to show on the placeholder (sanitized)
 * @returns {string} - Fallback image URL
 */
export function createPlaceholderUrl(text) {
  // Default text if none provided
  if (!text) return `${PLACEHOLDER_BASE_URL}/fe90ea/ffffff?text=Image`;
  
  // Sanitize and truncate text
  const sanitizedText = text
    .replace(/[^\w\s-]/g, '')
    .trim()
    .substring(0, 20);
  
  // Get color from text for consistency
  const bgColor = getHashColor(text).replace('#', '');
  
  // Create placeholder URL
  return `${PLACEHOLDER_BASE_URL}/${bgColor}/ffffff?text=${encodeURIComponent(sanitizedText)}`;
}

/**
 * Get proxied URL for external image sources
 * @param {string} url - Original image URL
 * @returns {string} - Proxied URL if applicable
 */
export function getProxiedUrl(url) {
  if (!url) return null;
  
  try {
    // Check if it's a Gumroad URL 
    if (url.includes('public-files.gumroad.com')) {
      // In development, use a direct URL to avoid CORS issues
      const isDevelopment = process.env.NODE_ENV === 'development';
      
      if (isDevelopment) {
        // Use the original URL in development
        return url;
      } else {
        // In production, use our Nginx proxy
        return url.replace('https://public-files.gumroad.com/', IMAGE_PROXY_PATH);
      }
    }
    
    // Return original URL for non-Gumroad URLs
    return url;
  } catch (error) {
    console.warn('Error processing URL for proxy:', error);
    return url;
  }
}

/**
 * Process a batch of products to add image URLs
 * @param {Array} products - Array of product objects
 * @returns {Array} - Processed products with image URLs
 */
export function processProductImages(products) {
  if (!products || !Array.isArray(products)) return [];
  
  return products.map(product => {
    if (!product) return product;
    
    // Add proxied URL if thumbnail exists
    const proxiedUrl = product.thumbnail_url ? getProxiedUrl(product.thumbnail_url) : null;
    
    // Generate placeholder for fallback
    const fallbackUrl = createPlaceholderUrl(product.name);
    
    return {
      ...product,
      proxied_thumbnail_url: proxiedUrl,
      fallback_url: fallbackUrl
    };
  });
}

/**
 * Preload a single image and return a promise
 * @param {string} url - Image URL to preload
 * @param {boolean} highPriority - Whether this is a high priority image
 * @returns {Promise} - Promise that resolves with loading status
 */
export function preloadImage(url, highPriority = false) {
  return new Promise(resolve => {
    if (!url) {
      resolve({ url, success: false });
      return;
    }
    
    const img = new Image();
    
    // Set high priority if supported
    if (highPriority && 'fetchpriority' in HTMLImageElement.prototype) {
      img.fetchpriority = 'high';
    }
    
    img.onload = () => resolve({ url, success: true });
    img.onerror = () => resolve({ url, success: false });
    img.src = url;
  });
}

/**
 * Simple function to prioritize loading top images first
 * @param {Array} results - The search results array
 */
export function prioritizeTopImages(results) {
  if (!results || results.length === 0) return;
  
  // Short timeout to ensure DOM is updated
  setTimeout(() => {
    // Get all product cards
    const productCards = document.querySelectorAll('.product-card');
    
    // Top priority images (first 10)
    const topCards = Array.from(productCards).slice(0, 10);
    
    // First, load the top 10 images immediately
    topCards.forEach(card => {
      const img = card.querySelector('img');
      if (img && img.getAttribute('loading') !== 'eager') {
        img.setAttribute('loading', 'eager');
        img.setAttribute('fetchpriority', 'high');
      }
    });
    
    // After 2 seconds, start loading other images
    setTimeout(() => {
      const remainingCards = Array.from(productCards).slice(10);
      remainingCards.forEach(card => {
        const img = card.querySelector('img');
        if (img) {
          // Ensure the image isn't being deferred by browser
          img.setAttribute('loading', 'eager');
        }
      });
    }, 2000);
  }, 100);
}

/**
 * Simplified function to preload high priority images
 * @param {Array} urls - Array of image URLs to preload
 * @param {boolean} highPriority - Whether these are high priority images
 * @returns {Promise} - Promise that resolves when preloading is complete
 */
export function preloadHighPriorityImages(urls, highPriority = true) {
  if (!urls || !Array.isArray(urls) || urls.length === 0) {
    return Promise.resolve([]);
  }
  
  // Only preload the top few images to avoid network congestion
  const imagesToPreload = urls.slice(0, highPriority ? 5 : 3);
  
  return Promise.all(
    imagesToPreload.map(url => preloadImage(url, highPriority))
  );
}

/**
 * Backward compatibility for preloadImages (redirects to preloadHighPriorityImages)
 * @param {Array} urls - Array of image URLs to preload
 * @param {boolean} highPriority - Whether these are high priority images
 * @returns {Promise} - Promise that resolves when preloading is complete
 */
export function preloadImages(urls, highPriority = false) {
  // Call our new function but keep the old API
  console.log('preloadImages is deprecated, use preloadHighPriorityImages instead');
  return preloadHighPriorityImages(urls, highPriority);
}

export default {
  createPlaceholderUrl,
  getProxiedUrl,
  processProductImages,
  preloadImage,
  preloadHighPriorityImages,
  preloadImages,
  prioritizeTopImages
};