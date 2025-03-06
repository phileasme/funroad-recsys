// services/api.js
import axios from 'axios';
import { processProductImages, preloadImages } from './imageService';

// Get the API base URL from environment or use default
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

// Create axios instance with defaults
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

/**
 * Enhanced search function that processes and preloads images
 * @param {string} profile - Search profile to use
 * @param {string} query - Search query
 * @param {number} k - Number of results to return
 * @param {number} numCandidates - Number of candidates to consider
 * @param {boolean} preloadImages - Whether to preload images (default true)
 * @returns {Promise<Object>} - Search results with processed image URLs
 */
export const searchProducts = async (profile, query, k = 50, numCandidates = 100, shouldPreload = true) => {
  console.log(`Calling ${profile} with query: ${query}`);
  
  try {
    // Make sure we're using the exact endpoint name from the backend
    const endpoint = profile;
    
    const payload = {
      query,
      k,
      num_candidates: numCandidates
    };
    
    console.log('Request payload:', payload);
    
    // Make the API request
    const response = await apiClient.post(`/${endpoint}`, payload);
    console.log('Search response:', response.data);
    
    // Process the results to include proxied image URLs
    if (response.data && response.data.results && Array.isArray(response.data.results)) {
      try {
        // Process URLs safely
        const processedResults = processProductImages(response.data.results);
        
        // Update the response data
        response.data.results = processedResults;
        
        // Optionally preload images in the background
        if (shouldPreload && processedResults.length > 0) {
          // Extract image URLs
          const imageUrls = processedResults
            .filter(product => product.proxied_thumbnail_url)
            .map(product => product.proxied_thumbnail_url);
          
          // Preload in the background without awaiting
          if (imageUrls.length > 0) {
            preloadImages(imageUrls)
              .catch(error => console.warn('Image preloading error:', error));
          }
        }
      } catch (processingError) {
        // If processing fails, log error but return original results
        console.error('Error processing search results images:', processingError);
      }
    }
    
    return response.data;
  } catch (error) {
    console.error('Search error:', error.response || error);
    throw error;
  }
};

/**
 * Get similar products based on a product
 * @param {string} description - Product description
 * @param {string} name - Product name
 * @param {string} id - Product ID for getting embeddings
 * @param {number} k - Number of results to return
 * @returns {Promise<Object>} - Similar products results
 */
export const getSimilarProducts = async (description, name, id, k = 10) => {
  try {
    console.log(`Getting similar products for ID: ${id || 'none'}`);
    
    // Create a query from either description or name
    const query = description 
      ? description.substring(0, 100) 
      : (name ? name.substring(0, 100) : '');
    
    const payload = {
      query: query,
      k: k,
      num_candidates: 100
    };
    
    // Only add the ID if it exists
    if (id) {
      payload.id = id;
    }
    
    console.log('Similar products payload:', payload);
    
    const response = await apiClient.post('/similar_vision', payload);
    console.log('Similar products response:', response.data);
    
    // Process the results to include proxied image URLs
    if (response.data && response.data.results && Array.isArray(response.data.results)) {
      try {
        // Process URLs safely
        const processedResults = processProductImages(response.data.results);
        
        // Update the response data
        response.data.results = processedResults;
        
        // Preload images in the background
        const imageUrls = processedResults
          .filter(product => product.proxied_thumbnail_url)
          .map(product => product.proxied_thumbnail_url);
        
        // Preload in the background without awaiting
        if (imageUrls.length > 0) {
          preloadImages(imageUrls)
            .catch(error => console.warn('Image preloading error:', error));
        }
      } catch (processingError) {
        // If processing fails, log error but return original results
        console.error('Error processing similar products images:', processingError);
      }
    }
    
    return response.data;
  } catch (error) {
    console.error('Error fetching similar products:', error.response || error);
    throw error;
  }
};

export default {
  searchProducts,
  getSimilarProducts
};