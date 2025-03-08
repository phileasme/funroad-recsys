/**
 * Service for handling images with improved loading and caching
 */

// In-memory cache for image status
const imageStatusCache = new Map();



const bgColors = ['212121', '4a4a4a', '6b6b6b', '444', '333', '555', '3c5a2d', '7f5e6b', '324d56', '742d1e'];
const textColors = ['ffffff', 'f0f0f0', 'eeeeee', 'dddddd', 'cccccc'];



// Constants
const IMAGE_PROXY_PATH = '/image-proxy/';
    
const IMAGE_LOAD_TIMEOUT = 3000;

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
  const defaultText = 'Loading...';
  

  const bgColor = bgColors[Math.floor(Math.random() * bgColors.length)];
  const textColor = textColors[Math.floor(Math.random() * textColors.length)];
  const FALLBACK_IMAGE_BASE = `https://placehold.co/600x400/${bgColor}/${textColor}?text=`;

  // Handle null/undefined text
  if (!text) {
    return `${FALLBACK_IMAGE_BASE}${encodeURIComponent(defaultText)}`;
  }
  
  try {
    // Try to sanitize and encode the text
    // First remove any problematic characters and limit length
    const sanitizedText = text
      .substring(0, 25)                  // Limit length
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
  if (!products || !products.length) {
    console.log("No products to process");
    return [];
  }
  
  console.log(`Processing ${products.length} products`);
  
  return products.map(product => {
    if (!product) return product;
    
    try {
      // Process each product safely
      let proxiedUrl = null;
      let fallbackUrl = null;
      
      // Create proxied URL if thumbnail exists
      if (product.thumbnail_url) {
        try {
          proxiedUrl = getProxiedImageUrl(product.thumbnail_url);
        } catch (err) {
          console.warn(`Failed to proxy URL for product ${product.id || 'unknown'}:`, err);
        }
      }
      
      // Create fallback URL
      try {
        fallbackUrl = createFallbackImageUrl(product.name);
      } catch (err) {
        console.warn(`Failed to create fallback URL for product ${product.id || 'unknown'}:`, err);
        fallbackUrl = `https://placehold.co/600x400?text=Image`;
      }
      
      // Create a new object to avoid mutating the original
      return {
        ...product,
        proxied_thumbnail_url: proxiedUrl,
        fallback_url: fallbackUrl
      };
    } catch (error) {
      // If processing fails completely, return the original product
      console.error('Error processing product image:', error);
      return product;
    }
  });
};

export default {
  getProxiedImageUrl,
  createFallbackImageUrl,
  preloadImage,
  preloadImages,
  processProductImages
};