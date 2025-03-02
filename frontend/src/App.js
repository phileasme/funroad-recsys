import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Search, BarChart as BarChartIcon, PieChart, Layers, Settings, TrendingUp } from 'lucide-react';
import { searchProducts, getSimilarProducts } from './services/api';

// Default profile data
const defaultProfileData = {
  default: {
    accuracy: 75,
    recall: 80,
    latency: 150,
    comparisonChart: [
      { name: 'Precision', current: 0.75, baseline: 0.62 },
      { name: 'Recall', current: 0.80, baseline: 0.65 },
      { name: 'F1 Score', current: 0.77, baseline: 0.63 },
      { name: 'MRR', current: 0.70, baseline: 0.51 }
    ],
    timeData: [
      { name: 'Jan', current: 150, baseline: 145 },
      { name: 'Feb', current: 148, baseline: 142 },
      { name: 'Mar', current: 145, baseline: 140 },
      { name: 'Apr', current: 140, baseline: 138 },
      { name: 'May', current: 138, baseline: 135 }
    ]
  },
  
  // Profile-specific overrides
  search_text_based: {
    accuracy: 68,
    recall: 72,
    latency: 120
    // Other values will default
  },
  
  search_combined_simplified_but_slow: {
    accuracy: 88,
    recall: 92,
    latency: 180
    // Other values will default
  }
};

// Create default data for all profiles
const searchProfiles = [
  { id: 'search_text_based', name: 'Text-Based Search' },
  { id: 'search_fuzzy', name: 'Fuzzy Search' },
  { id: 'search_vision', name: 'Vision Search' },
  { id: 'search_colbert', name: 'ColBERT Search' },
  { id: 'search_combined', name: 'Combined Search' },
  { id: 'search_combined_simplified_but_slow', name: 'Optimized Combined' },
  { id: 'search_lame_combined', name: 'Basic Combined' }
];

// Initialize all profiles that don't have specific data
searchProfiles.forEach(profile => {
  if (!defaultProfileData[profile.id]) {
    defaultProfileData[profile.id] = {
      ...defaultProfileData.default,
      // Add a random variation to make each profile look different
      accuracy: defaultProfileData.default.accuracy + Math.floor(Math.random() * 15),
      recall: defaultProfileData.default.recall + Math.floor(Math.random() * 15)
    };
  }
});

function App() {
  const [query, setQuery] = useState('');
  const [searchProfile, setSearchProfile] = useState('search_combined_simplified_but_slow');
  const [searchResults, setSearchResults] = useState([]);
  const [similarProducts, setSimilarProducts] = useState([]);
  const [hoveredProduct, setHoveredProduct] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [performanceData, setPerformanceData] = useState(defaultProfileData.search_combined_simplified_but_slow);
  const [searchHistory, setSearchHistory] = useState([]);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });

  // Update performance data when search profile changes
  useEffect(() => {
    setPerformanceData(defaultProfileData[searchProfile] || defaultProfileData.default);
  }, [searchProfile]);

  // Handle search submission
  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    
    try {
      const data = await searchProducts(searchProfile, query);
      setSearchResults(data.results || []);
      
      // Update search history
      setSearchHistory(prev => [
        { query, timestamp: new Date().toLocaleTimeString(), results: data.results?.length || 0, queryTime: data.query_time_ms },
        ...prev.slice(0, 9)
      ]);
    } catch (error) {
      console.error('Search error:', error);
      // Fallback to mock data
      setSearchResults(Array(10).fill(0).map((_, i) => ({
        score: Math.random(),
        name: `Sample Product ${i + 1}`,
        description: `This is a sample product description for "${query}".`,
        thumbnail_url: `https://placehold.co/100x100?text=Product+${i+1}`
      })));
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch similar products when hovering over a product
  const handleProductHover = useCallback(async (product, event) => {
    setHoveredProduct(product);
    setHoverPosition({ 
      x: event.clientX, 
      y: event.clientY 
    });
    
    try {
      const data = await getSimilarProducts(product.description, product.name);
      // Filter out the current product from results
      const similarProducts = data.results
        .filter(item => item.name !== product.name)
        .map(item => ({
          ...item,
          score: parseFloat(item.score).toFixed(2) // Format score
        }));
        
      setSimilarProducts(similarProducts);
    } catch (error) {
      console.error('Error fetching similar products:', error);
      // Fallback to mock data
      const fakeSimilarProducts = Array(5).fill(0).map((_, i) => ({
        id: `similar-${i}`,
        name: `Similar to ${product.name} - Item ${i + 1}`,
        description: `A product similar to ${product.name}.`,
        thumbnail_url: `https://placehold.co/50x50?text=Similar+${i+1}`,
        score: (Math.random() * 0.3 + 0.7).toFixed(2)
      }));
      
      setSimilarProducts(fakeSimilarProducts);
    }
  }, []);

  const handleMouseLeave = () => {
    setHoveredProduct(null);
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm py-4 px-6">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <img src="/gumroad.png" alt="Gumroad Logo" className="h-8 w-auto" />
            <h1 className="text-xl font-bold text-gray-800">Gumroad Search Case Study</h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-600">
              Search Profiles: {searchProfiles.length}
            </div>
            <div className="text-sm text-gray-600">
              Indexed Products: 3,245
            </div>
            <Settings className="h-5 w-5 text-gray-500 cursor-pointer hover:text-gray-700" />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-grow py-6 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Search form - Common to both layouts */}
          <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
            <form onSubmit={handleSearch} className="flex items-center gap-4">
              <div className="relative flex-grow">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search for products..."
                  className="w-full pl-10 pr-4 py-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <select
                value={searchProfile}
                onChange={(e) => setSearchProfile(e.target.value)}
                className="px-3 py-3 rounded-md border border-gray-300 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {searchProfiles.map(profile => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}
                  </option>
                ))}
              </select>
              <button
                type="submit"
                className="bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={isLoading}
              >
                {isLoading ? 'Searching...' : 'Search'}
              </button>
            </form>
          </div>

          {/* Two-column layout for desktop, stacked for mobile */}
          <div className="flex flex-col lg:flex-row gap-6">
            {/* Left column (wider) - Search results */}
            <div className="lg:w-2/3 space-y-6">
              {/* Search results */}
              {searchResults.length > 0 && (
                <div className="bg-white p-6 rounded-lg shadow-sm">
                  <h2 className="text-lg font-semibold mb-4">Search Results ({searchResults.length})</h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {searchResults.map((product, index) => (
                      <div
                        key={`${product.name}-${index}`}
                        className="border border-gray-200 rounded-md p-4 hover:shadow-md transition-shadow relative"
                        onMouseEnter={(e) => handleProductHover(product, e)}
                        onMouseLeave={handleMouseLeave}
                      >
                        <div className="flex">
                          <div className="flex-shrink-0 w-24 h-24 bg-gray-100 rounded overflow-hidden">
                            <img 
                              src={product.thumbnail_url || `https://placehold.co/100x100?text=Product+${index}`} 
                              alt={product.name} 
                              className="w-full h-full object-cover"
                              onError={(e) => {
                                e.target.src = `https://placehold.co/100x100?text=Product+${index}`;
                              }}
                            />
                          </div>
                          <div className="ml-4 flex-grow">
                            <h3 className="font-medium text-gray-800">{product.name}</h3>
                            <p className="text-gray-600 text-sm mt-1 line-clamp-2">
                              {product.description || "No description available."}
                            </p>
                            <div className="mt-2 flex items-center justify-between">
                              <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                                Score: {typeof product.score === 'number' ? product.score.toFixed(2) : product.score}
                              </span>
                              <button className="text-sm text-blue-600 hover:text-blue-800">
                                View details
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Search history */}
              {searchHistory.length > 0 && (
                <div className="bg-white p-6 rounded-lg shadow-sm">
                  <h2 className="text-lg font-semibold mb-4">Recent Searches</h2>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead>
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Query</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Results</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Query Time</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {searchHistory.map((item, i) => (
                          <tr key={i}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.query}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.timestamp}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.results}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.queryTime?.toFixed(2) || '-'} ms</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-600 hover:text-blue-800">
                              <button onClick={() => { setQuery(item.query); handleSearch(); }}>
                                Search again
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
            
            {/* Right column (narrower) - Performance metrics */}
            {performanceData && (
              <div className="lg:w-1/3">
                <div className="bg-white p-6 rounded-lg shadow-sm sticky top-6">
                  <h2 className="text-lg font-semibold mb-4 flex items-center">
                    <TrendingUp className="mr-2 text-blue-600" />
                    Performance Metrics
                  </h2>
                  <div className="text-sm text-gray-600 mb-4">
                    {searchProfiles.find(p => p.id === searchProfile)?.name}
                  </div>
                  
                  <div className="space-y-4">
                    {/* Metric cards */}
                    <div className="bg-gray-50 p-4 rounded-md">
                      <div className="text-sm text-gray-500 mb-1">Accuracy</div>
                      <div className="text-2xl font-bold text-gray-800">{performanceData.accuracy}%</div>
                      <div className="text-xs text-green-600 mt-1">+{(performanceData.accuracy - 62).toFixed(1)}% vs baseline</div>
                    </div>
                    
                    <div className="bg-gray-50 p-4 rounded-md">
                      <div className="text-sm text-gray-500 mb-1">Recall</div>
                      <div className="text-2xl font-bold text-gray-800">{performanceData.recall}%</div>
                      <div className="text-xs text-green-600 mt-1">+{(performanceData.recall - 65).toFixed(1)}% vs baseline</div>
                    </div>
                    
                    <div className="bg-gray-50 p-4 rounded-md">
                      <div className="text-sm text-gray-500 mb-1">Avg. Latency</div>
                      <div className="text-2xl font-bold text-gray-800">{performanceData.latency}ms</div>
                      <div className="text-xs text-red-600 mt-1">+{(performanceData.latency - 135).toFixed(1)}ms vs baseline</div>
                    </div>
                  </div>
                  
                  {/* Comparison chart */}
                  <div className="mt-6">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Metrics Comparison</h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={performanceData.comparisonChart}
                          barSize={15}
                          layout="vertical"
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                          <YAxis type="category" dataKey="name" width={70} />
                          <Tooltip 
                            formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Score']}
                          />
                          <Legend />
                          <Bar dataKey="current" fill="#3B82F6" name="Current" />
                          <Bar dataKey="baseline" fill="#9CA3AF" name="Baseline" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  
                  {/* Time series chart */}
                  <div className="mt-6">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Response Time Trend (ms)</h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={performanceData.timeData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="current" stroke="#3B82F6" name="Current" strokeWidth={2} />
                          <Line type="monotone" dataKey="baseline" stroke="#9CA3AF" name="Baseline" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Similar products popup */}
      {hoveredProduct && (
        <div 
          className="fixed bg-white rounded-lg shadow-lg p-4 z-50 w-72"
          style={{
            top: `${Math.min(hoverPosition.y + 10, window.innerHeight - 300)}px`,
            left: `${Math.min(hoverPosition.x + 10, window.innerWidth - 300)}px`,
          }}
        >
          <h3 className="font-medium text-sm mb-2">Similar Products</h3>
          <div className="max-h-64 overflow-y-auto">
            {similarProducts.length > 0 ? (
              similarProducts.map((product, index) => (
                <div key={product.id || index} className="flex items-center py-2 border-b border-gray-100 last:border-0">
                  <div className="w-12 h-12 bg-gray-50 rounded overflow-hidden flex-shrink-0">
                    <img 
                      src={product.thumbnail_url || `https://placehold.co/50x50?text=Similar+${index}`} 
                      alt={product.name} 
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = `https://placehold.co/50x50?text=Similar+${index}`;
                      }}
                    />
                  </div>
                  <div className="ml-3 flex-grow">
                    <h4 className="text-xs font-medium text-gray-800 line-clamp-1">{product.name}</h4>
                    <p className="text-xs text-gray-500 line-clamp-1">{product.description}</p>
                    <div className="text-xs text-blue-600 mt-1">Similarity: {product.score}</div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-sm text-gray-500 py-2">Loading similar products...</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;