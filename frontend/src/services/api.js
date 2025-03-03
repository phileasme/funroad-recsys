// services/api.js
import axios from 'axios';

// Get the API base URL from environment or use default
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Search for products using the specified search profile
 * @param {string} profile - The search profile to use
 * @param {string} query - The search query
 * @param {number} k - Number of results to return
 * @param {number} numCandidates - Number of candidates to consider
 * @returns {Promise<Object>} - The search results
 */
export const searchProducts = async (profile, query, k = 50, numCandidates = 100) => {
  console.log(`Calling ${profile} with query: ${query}`);
  
  try {
    // Make sure we're using the exact endpoint name from the backend
    const endpoint = profile;
    
    const payload = {
      query,
      k,
      num_candidates: numCandidates
      // id is not included here since it's optional and not needed for basic search
    };
    
    console.log('Request payload:', payload);
    
    // Log the complete URL being called
    console.log('API URL:', `${API_BASE_URL}/${endpoint}`);
    
    const response = await apiClient.post(`/${endpoint}`, payload);
    console.log('Search response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Search error:', error.response || error);
    // If we get a 422 error, log the validation errors from FastAPI
    if (error.response && error.response.status === 422) {
      console.error('Validation error details:', error.response.data);
    }
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
    return response.data;
  } catch (error) {
    console.error('Error fetching similar products:', error.response || error);
    throw error;
  }
};