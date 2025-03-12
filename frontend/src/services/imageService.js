// src/services/imageService.js
/**
 * Enhanced image service with better loading, caching, and fallback strategy
 */

// In-memory LRU cache for image status - limited to 500 entries to avoid memory leaks
class LRUCache {
  constructor(maxSize = 500) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  get(key) {
    if (!this.cache.has(key)) return undefined;
    
    // Access refreshes position in the cache
    const value = this.cache.get(key);
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  set(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Remove oldest entry (first key in map)
      this.cache.delete(this.cache.keys().next().value);
    }
    this.cache.set(key, value);
  }

  has(key) {
    return this.cache.has(key);
  }

  clear() {
    this.cache.clear();
  }
}

// Create the cache instance
const imageStatusCache = new LRUCache(500);

// Pre-defined placeholder colors - optimized shorter list
const BG_COLORS = ['212121', '4a4a4a', '6b6b6b', '444', '555', 'fe90ea'];
const TEXT_COLORS = ['ffffff', 'f0f0f0', 'eeeeee'];

// Constants
const IMAGE_PROXY_PATH = '/image-proxy/';
const IMAGE_LOAD_TIMEOUT = 2500; // Reduced timeout
const PLACEHOLDER_BASE = 'https://placehold.co';

// Store preloaded URLs to avoid duplicate requests
const preloadedUrls = new Set();

/**
 * Get a color from the arrays using a hash of the input string
 * This ensures the same string always gets the same color
 */
const getConsistentColor = (str) => {
  if (!str) return { bg: BG_COLORS[0], text: TEXT_COLORS[0] };
  
  // Simple hash function
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash = hash & hash; // Convert to 32bit integer
  }
  
  // Use absolute value and modulo to get indices
  const bgIndex = Math.abs(hash) % BG_COLORS.length;
  const textIndex = Math.abs(hash >> 4) % TEXT_COLORS.length; // Use bit shift for different index
  
  return {
    bg: BG_COLORS[bgIndex],
    text: TEXT_COLORS[textIndex]
  };
};

/**
 * Rewrite a Gumroad URL to use our proxy
 * @param {string} url - Original image URL
 * @returns {string} - Proxied URL if it's a Gumroad URL, original otherwise
 */
export const getProxiedImageUrl = (url) => {
  // Handle empty/invalid URLs
  if (!url) return null;
  
  try {
    // Basic URL validation
    if (typeof url !== 'string' || !url.includes('http')) {
      return url;
    }
    
    // Check if it's a Gumroad URL 
    if (url.includes('public-files.gumroad.com')) {
      // In development, use a direct URL to avoid CORS issues
      const isDevelopment = process.env.NODE_ENV === 'development';
      
      if (isDevelopment) {
        // Use the original URL in development
        return url;
      } else {
        // In production, use our Nginx proxy
        return url.replace('https://public-files.gumroad.com/', '/image-proxy/');
      }
    }
    
    // Return original URL for non-Gumroad URLs
    return url;
  } catch (error) {
    console.warn('Error processing URL for proxy:', error);
    return url;
  }
};

/**
 * Create a fallback image URL with encoded text
 * @param {string} text - Text to show on the placeholder image
 * @param {number} width - Width of the placeholder
 * @param {number} height - Height of the placeholder
 * @returns {string} - Fallback image URL
 */
export const createFallbackImageUrl = (text, width = 600, height = 400) => {
  // Default text if none provided
  const defaultText = 'Image';
  
  // Get consistent colors based on the text
  const { bg, text: textColor } = getConsistentColor(text);
  
  // Handle null/undefined text
  if (!text) {
    return `${PLACEHOLDER_BASE}/${width}x${height}/${bg}/${textColor}?text=${defaultText}`;
  }
  
  try {
    // Try to sanitize and encode the text
    // Remove any problematic characters and limit length
    const sanitizedText = text
      .substring(0, 20)                  // Limit length
      .replace(/[^\w\s-]/g, '')          // Remove special characters
      .trim();                           // Remove whitespace
    
    // If sanitizing removed everything, use default
    const finalText = sanitizedText || defaultText;
    
    // Encode the sanitized text
    return `${PLACEHOLDER_BASE}/${width}x${height}/${bg}/${textColor}?text=${encodeURIComponent(finalText)}`;
  } catch (error) {
    // If any encoding error happens, use a simple fallback
    console.warn('Error creating fallback image URL:', error);
    return `${PLACEHOLDER_BASE}/${width}x${height}/${bg}/${textColor}?text=${defaultText}`;
  }
};

/**
 * Create a data URL for a colored rectangle to use as a placeholder
 * This is faster than loading an external placeholder image
 * @param {string} bgColor - Background color hex (without #)
 * @param {number} width - Width of the placeholder
 * @param {number} height - Height of the placeholder
 * @returns {string} - Data URL for a colored SVG rectangle
 */
export const createInlinePlaceholder = (text, width = 600, height = 400) => {
  const { bg } = getConsistentColor(text);
  const hexColor = `#${bg}`;
  
  // Simple SVG rectangle
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="100%" height="100%" fill="${hexColor}" />
  </svg>`;
  
  // Convert to a data URL
  return `data:image/svg+xml;base64,${btoa(svg)}`;
};

/**
 * Preload an image and return a promise that resolves when loaded
 * @param {string} url - Image URL to preload
 * @param {boolean} useProxy - Whether to use the proxy for Gumroad URLs
 * @returns {Promise} - Promise that resolves with success/failure status
 */
export const preloadImage = (url, useProxy = true) => {
  return new Promise((resolve) => {
    if (!url) {
      resolve({ success: false, originalUrl: url });
      return;
    }
    
    // Check if already preloaded
    if (preloadedUrls.has(url)) {
      resolve({ success: true, originalUrl: url, alreadyPreloaded: true });
      return;
    }
    
    // Check cache first
    if (imageStatusCache.has(url)) {
      resolve(imageStatusCache.get(url));
      return;
    }
    
    // Process URL if needed
    const processedUrl = useProxy ? getProxiedImageUrl(url) : url;
    
    // Create image to preload
    const img = new Image();
    let resolved = false;
    
    // Set timeout to avoid waiting too long
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        const result = { success: false, originalUrl: url, processedUrl };
        imageStatusCache.set(url, result);
        resolve(result);
      }
    }, IMAGE_LOAD_TIMEOUT);
    
    // Success handler
    img.onload = () => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        const result = { 
          success: true, 
          originalUrl: url, 
          processedUrl,
          width: img.width,
          height: img.height,
          aspectRatio: img.width / img.height
        };
        imageStatusCache.set(url, result);
        preloadedUrls.add(url); // Mark as preloaded
        resolve(result);
      }
    };
    
    // Error handler
    img.onerror = () => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        const result = { success: false, originalUrl: url, processedUrl };
        imageStatusCache.set(url, result);
        resolve(result);
      }
    };
    
    // Start loading
    img.src = processedUrl;
    
    // Add decoding attribute for better performance
    img.decoding = 'async';
  });
};

/**
 * Batch preload multiple images with priority
 * @param {Array<string>} urls - Array of image URLs to preload
 * @param {boolean} useProxy - Whether to use the proxy
 * @returns {Promise} - Promise that resolves when all images are processed
 */
export const preloadImages = async (urls, useProxy = true) => {
  if (!urls || !urls.length) return [];
  
  // Limit batch size to avoid too many simultaneous requests
  const BATCH_SIZE = 8;
  const uniqueUrls = [...new Set(urls)].filter(url => !preloadedUrls.has(url));
  
  if (uniqueUrls.length === 0) {
    return [];
  }
  
  // Process in batches
  const results = [];
  for (let i = 0; i < uniqueUrls.length; i += BATCH_SIZE) {
    const batch = uniqueUrls.slice(i, i + BATCH_SIZE);
    const batchResults = await Promise.all(batch.map(url => preloadImage(url, useProxy)));
    results.push(...batchResults);
  }
  
  return results;
};

/**
 * Prioritize images for preloading (visible first, then others)
 * @param {Array<Object>} products - Products to prioritize
 * @param {number} visibleCount - Number of initially visible products
 * @returns {Array<string>} - Prioritized array of image URLs
 */
export const getPrioritizedImageUrls = (products, visibleCount = 12) => {
  if (!products || !products.length) return [];
  
  // Split into visible and non-visible
  const visible = products.slice(0, visibleCount);
  const nonVisible = products.slice(visibleCount);
  
  // Get URLs from visible products first
  const visibleUrls = visible
    .map(p => p.thumbnail_url)
    .filter(Boolean);
    
  // Then get URLs from non-visible products
  const nonVisibleUrls = nonVisible
    .map(p => p.thumbnail_url)
    .filter(Boolean);
    
  // Return combined array with visible URLs first
  return [...visibleUrls, ...nonVisibleUrls];
};

/**
 * Process product objects to add proxied image URLs
 * @param {Array<Object>} products - Product objects with thumbnail_url
 * @returns {Array<Object>} - Products with processed URLs
 */
export const processProductImages = (products) => {
  if (!products || !products.length) {
    return [];
  }
  
  return products.map(product => {
    if (!product) return product;
    
    try {
      // Process each product safely
      let proxiedUrl = null;
      let fallbackUrl = null;
      let inlinePlaceholder = null;
      
      // Create proxied URL if thumbnail exists
      if (product.thumbnail_url) {
        try {
          proxiedUrl = getProxiedImageUrl(product.thumbnail_url);
        } catch (err) {
          console.warn(`Failed to proxy URL for product ${product.id || 'unknown'}`);
        }
      }
      
      // Create fallback URL and inline placeholder
      try {
        fallbackUrl = createFallbackImageUrl(product.name);
        inlinePlaceholder = createInlinePlaceholder(product.name);
      } catch (err) {
        fallbackUrl = `${PLACEHOLDER_BASE}/600x400/212121/ffffff?text=Image`;
        inlinePlaceholder = createInlinePlaceholder('Image');
      }
      
      // Return enhanced product object
      return {
        ...product,
        proxied_thumbnail_url: proxiedUrl,
        fallback_url: fallbackUrl,
        inline_placeholder: inlinePlaceholder,
        // Check if this image is already cached
        is_image_cached: proxiedUrl ? imageStatusCache.has(proxiedUrl) : false
      };
    } catch (error) {
      // If processing fails, return original product
      return product;
    }
  });
};

// Clear cache and state when leaving the page
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    imageStatusCache.clear();
    preloadedUrls.clear();
  });
}

export default {
  getProxiedImageUrl,
  createFallbackImageUrl,
  createInlinePlaceholder,
  preloadImage,
  preloadImages,
  getPrioritizedImageUrls,
  processProductImages
};