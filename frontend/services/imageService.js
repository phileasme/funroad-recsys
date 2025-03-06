// services/imageService.js

/**
 * Service for handling images with improved loading and caching
 */

// In-memory cache for image status
const imageStatusCache = new Map();

// Constants
const IMAGE_PROXY_PATH = '/image-proxy/';
const FALLBACK_IMAGE_BASE = 'https://placehold.co/600x400?text=';
const IMAGE_LOAD_TIMEOUT = 8000; // 8 seconds timeout

/**
 * Rewrite a Gumroad URL to use our proxy
 * @param {string} url - Original image URL
 * @returns {string} - Proxied URL if it's a Gumroad URL, original otherwise
 */
export const getProxiedImageUrl = (url) => {
  if (!url) return null;
  
  // Check if it's a Gumroad URL
  if (url.includes('public-files.gumroad.com')) {
    // Replace the domain with our proxy path
    return url.replace('https://public-files.gumroad.com/', IMAGE_PROXY_PATH);
  }
  
  // Return original URL for non-Gumroad URLs
  return url;
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
  const defaultText = 'Loading...';
  
  // Handle null/undefined text
  if (!text) {
    return `${FALLBACK_IMAGE_BASE}${encodeURIComponent(defaultText)}`;
  }
  
  try {
    // Try to sanitize and encode the text
    // First remove any problematic characters and limit length
    const sanitizedText = String(text)
      .substring(0, 20)                  // Limit length
      .replace(/[^\w\s-]/g, '')          // Remove special characters
      .trim();                           // Remove leading/trailing whitespace
    
    // If sanitizing removed everything, use default
    const finalText = sanitizedText || defaultText;
    
    // Encode the sanitized text
    return `${FALLBACK_IMAGE_BASE}${encodeURIComponent(finalText)}`;
  } catch (error) {
    // If any encoding error happens, use a simple fallback
    console.warn('Error creating fallback image URL:', error);
    return `${FALLBACK_IMAGE_BASE}Image`;
  }
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
  });
};

/**
 * Batch preload multiple images
 * @param {Array<string>} urls - Array of image URLs to preload
 * @param {boolean} useProxy - Whether to use the proxy
 * @returns {Promise} - Promise that resolves when all images are processed
 */
export const preloadImages = async (urls, useProxy = true) => {
  if (!urls || !urls.length) return [];
  
  // Use Promise.allSettled to prevent one failure from blocking others
  return Promise.allSettled(
    urls.map(url => preloadImage(url, useProxy))
  );
};

/**
 * Process product objects to add proxied image URLs
 * @param {Array<Object>} products - Product objects with thumbnail_url
 * @returns {Array<Object>} - Products with processed URLs
 */
export const processProductImages = (products) => {
  if (!products || !products.length) return [];
  
  return products.map(product => {
    if (!product) return product;
    
    // Create a new object to avoid mutating the original
    return {
      ...product,
      // Add these new properties
      proxied_thumbnail_url: product.thumbnail_url ? getProxiedImageUrl(product.thumbnail_url) : null,
      fallback_url: createFallbackImageUrl(product.name)
    };
  });
};

export default {
  getProxiedImageUrl,
  createFallbackImageUrl,
  preloadImage,
  preloadImages,
  processProductImages
};