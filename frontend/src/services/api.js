import axios from 'axios';

// Get the API base URL from environment or use default
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const searchProducts = async (profile, query, k = 10, numCandidates = 100) => {
  try {
    const response = await apiClient.post(`/${profile}`, {
      query,
      k,
      num_candidates: numCandidates
    });
    return response.data;
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
};

export const getSimilarProducts = async (description, name, k = 10) => {
  try {
    const query = description ? description.substring(0, 100) : name;
    const response = await apiClient.post('/search_text_based', {
      query,
      k,
      num_candidates: 50
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching similar products:', error);
    throw error;
  }
};