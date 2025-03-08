// services/api.js
import axios from 'axios';
import { processProductImages, preloadImages } from './imageService';


const isDevelopment = process.env.NODE_ENV === 'development';

// In development, point directly to the API server
const API_BASE_URL = isDevelopment 
  ? 'http://localhost:8000' 
  : '/api';


// Get the API base URL from environment or use default
// const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';
// Create axios instance with defaults
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});


// Add this function to test connectivity
export const testApiConnection = async () => {
  try {
    const response = await apiClient.get('/health');
    console.log('API connection successful:', response.data);
    return true;
  } catch (error) {
    console.error('API connection failed:', error);
    return false;
  }
};


// In your api.js file
export const searchProducts = async (profile, query, k = 50, numCandidates = 100, shouldPreload = true) => {
  console.log(`Calling ${profile} with query: ${query}`);
  
  try {
    const endpoint = profile;
    
    const payload = {
      query,
      k,
      num_candidates: numCandidates
    };
    
    console.log('Request payload:', payload);
    
    // Reduce timeout and add better error handling
    const response = await apiClient.post(`/${endpoint}`, payload, {
      timeout: 10000 // Reduce timeout to 10 seconds
    });
    
    console.log('Search response:', response.data);
    
    // Process the results as before...
    return response.data;
  } catch (error) {
    // Better error handling
    if (error.code === 'ECONNABORTED') {
      console.error('Search request timed out. Try reducing the number of results or candidates.');
    } else {
      console.error('Search error:', error.response || error);
    }
    
    // Return empty results instead of throwing
    return { results: [], query_time_ms: 0 };
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