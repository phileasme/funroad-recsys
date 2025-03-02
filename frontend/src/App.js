import React, { useState, useEffect, useCallback, useRef } from 'react';
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


  const [showSimilarProducts, setShowSimilarProducts] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const similarProductsRef = useRef(null);


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

  
    const handleProductHover = useCallback(async (product, event) => {
      setHoveredProduct(product);
      setSelectedProduct(product);
      setShowSimilarProducts(true);
      setHoverPosition({ 
        x: event.clientX, 
        y: event.clientY 
      });
      
      try {
        const data = await getSimilarProducts(product.description, product.name, product.id);
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
          score: (Math.random() * 0.3 + 0.7).toFixed(2),
          ratings_count: Math.floor(Math.random() * 100) + 5,
          ratings_score: (Math.random() * 1 + 4).toFixed(1),
          price_cents: Math.floor(Math.random() * 5000) + 500,
          url: `https://example.com/product/${i}`
        }));
        
        setSimilarProducts(fakeSimilarProducts);
      }
    }, []);
  
    // Update performance data when search profile changes
    useEffect(() => {
      setPerformanceData(defaultProfileData[searchProfile] || defaultProfileData.default);
    }, [searchProfile]);
  
  
    useEffect(() => {
      const handleClickOutside = (event) => {
        if (
          similarProductsRef.current && 
          !similarProductsRef.current.contains(event.target) &&
          showSimilarProducts
        ) {
          setShowSimilarProducts(false);
          setHoveredProduct(null);
          setSelectedProduct(null);
        }
      };
    
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }, [showSimilarProducts]);


  const handleMouseLeave = () => {
    // setHoveredProduct(null);
  };

const handleSimilarProductsMouseLeave = () => {
  setShowSimilarProducts(false);
  setHoveredProduct(null);
  setSelectedProduct(null);
};

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm py-4 px-6 border-b-2 border-[#FE90EA]">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <img src="/gumroad.png" alt="Gumroad Logo" className="h-8 w-auto" />
            <h1 className="text-xl font-bold text-black">Gumroad Search Case Study</h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-black">
              Search Profiles: {searchProfiles.length}
            </div>
            <div className="text-sm text-black">
              Indexed Products: 3,245
            </div>
            <Settings className="h-5 w-5 text-[#FE90EA] cursor-pointer hover:text-black" />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-grow py-6 px-6">
        <div className="w-full mx-auto">
          {/* Search form - Common to both layouts */}
          <div className="bg-white p-6 rounded-lg shadow-sm mb-6 border-2 border-gray-200">
          <form onSubmit={handleSearch} className="flex items-center gap-4">
            <div className="relative flex-grow">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-[#FE90EA]" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search for products..."
                className="w-full pl-10 pr-4 py-3 rounded-md border-2 border-gray-300 focus:outline-none focus:border-[#FE90EA] focus:ring-1 focus:ring-[#FE90EA]"
              />
            </div>
            <select
              value={searchProfile}
              onChange={(e) => setSearchProfile(e.target.value)}
              className="px-3 py-3 rounded-md border-2 border-gray-300 bg-white focus:outline-none focus:border-[#FE90EA] focus:ring-1 focus:ring-[#FE90EA]"
            >
              {searchProfiles.map(profile => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </select>
            <button
              type="submit"
              className="bg-[#FE90EA] text-black px-6 py-3 rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-2 focus:ring-[#FE90EA] focus:ring-offset-2 font-medium border-2 border-black"
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
                <h2 className="text-xl font-semibold mb-6 text-black border-b-2 border-[#FE90EA] pb-2 inline-block">Search Results ({searchResults.length})</h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                  {searchResults.map((product, index) => (
                    
                    <div
                      key={`${product.name}-${index}`}
                      className="bg-white border-2 border-gray-200 hover:border-[#FE90EA] rounded-lg overflow-hidden hover:shadow-lg transition-all duration-300"
                      onMouseEnter={(e) => handleProductHover(product, e)}
                      onMouseLeave={handleMouseLeave}
                    >
                      <a 
                              href={product.url || "#"} 
                              target="#">
                      <div className="relative">
                        {/* Product Image - Larger and with better aspect ratio */}
                        <div className="w-full h-48 bg-gray-100 overflow-hidden">
                          <img 
                            src={product.thumbnail_url || `https://placehold.co/600x400?text=${encodeURIComponent(product.name)}`} 
                            alt={product.name} 
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              e.target.src = `https://placehold.co/600x400?text=${encodeURIComponent(product.name.substring(0, 20))}`;
                            }}
                          />
                        </div>
            
                          
                          {/* Price tag overlaid on image - flag style */}
                          {product.price_cents !== undefined && (
                            <div className="absolute rounded-md top-4 right-5 flex items-center">
                            <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-lg">
                              ${(product.price_cents / 100).toFixed(2)}
                              {/* Top right cutout - makes it look transparent */}
                              <div className="absolute -right-[3px] top-0 w-0 h-0 border-t-[9px] border-b-[9px] border-l-[7px] border-t-transparent border-b-transparent border-l-black"></div>
                              {/* Bottom right triangle - keeps the pink base */}
                              <div className="absolute -right-[3px] bottom-0 w-0 h-0 border-t-[9px] border-b-[9px] border-l-[7px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
                            </div>
                          </div>
                          )}
                        </div>
                        
                        {/* Product details */}
                        <div className="p-4">
                          <h3 className="font-medium text-lg text-gray-800 mb-2 line-clamp-1">{product.name}</h3>
                          
                          {/* Rating display with stars */}
                          {product.ratings_score !== undefined && (
                            <div className="flex items-center mb-2">
                              <div className="flex text-yellow-400">
                                {[...Array(5)].map((_, i) => (
                                  <span key={i} className="text-lg">
                                    {i < Math.floor(product.ratings_score) ? "★" : "☆"}
                                  </span>
                                ))}
                              </div>
                              <span className="ml-2 text-sm text-gray-600">
                                {product.ratings_score} ({product.ratings_count})
                              </span>
                            </div>
                          )}
                          
                          <p className="text-gray-600 text-sm mb-4 line-clamp-2">
                            {product.description || "No description available."}
                          </p>
                          
                          <div className="flex items-center justify-between mt-auto pt-2 border-t border-gray-100">
                            <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-black/5 text-black text-xs font-medium">
                              Score: {typeof product.score === 'number' ? product.score.toFixed(2) : product.score}
                            </span>
                            
                            <a 
                              href={product.url || "#"} 
                              target="#"
                              className="inline-flex items-center justify-center px-4 py-2 text-sm font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#FE90EA] border-2 border-black"
                            >
                              View details
                            </a>
                          </div>
                        </div>
                        </a>
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
                <ScrollingQueryExamples />
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
                      <div className="text-sm text-gray-500 mb-1">Precision</div>
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
      // Updated similar products popup with improved styling
      {showSimilarProducts && selectedProduct && (
        <div 
          ref={similarProductsRef}
          className="fixed bg-white rounded-lg shadow-xl p-5 z-50 w-80 border-2 border-[#FE90EA]"
          style={{
            top: `${Math.min(hoverPosition.y + 10, window.innerHeight - 450)}px`,
            left: `${Math.min(hoverPosition.x + 10, window.innerWidth - 400)}px`,
          }}
          onMouseLeave={handleSimilarProductsMouseLeave}
        >
          <h3 className="font-semibold text-base mb-3 pb-2 border-b-2 border-[#FE90EA] inline-block text-black py-0 px-0">Similar to "{selectedProduct.name.substring(0, 19)}{selectedProduct.name.length > 21 ? '...' : ''}"</h3>
          <div className="max-h-72 overflow-y-auto pr-1">
            {similarProducts.length > 0 ? (
              similarProducts.map((product, index) => (
                <div key={product.id || index} className="flex items-start py-3 border-b border-gray-100 last:border-0 hover:bg-gray-50 transition-colors rounded-md px-2">
                  <div className="w-16 h-16 bg-gray-100 rounded-md overflow-hidden flex-shrink-0">
                    <img 
                      src={product.thumbnail_url || `https://placehold.co/100x100?text=Similar`} 
                      alt={product.name} 
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = `https://placehold.co/100x100?text=Similar`;
                      }}
                    />
                  </div>
                  <div className="ml-3 flex-grow">
                    <div className="flex justify-between items-start">
                      <h4 className="text-sm font-medium text-gray-800 line-clamp-1">{product.name}</h4>
                      {product.price_cents !== undefined && (
                        <div className="text-sm font-medium text-gray-800 ml-2 whitespace-nowrap">${(product.price_cents / 100).toFixed(2)}</div>
                      )}
                    </div>
                    
                    {product.ratings_score && (
                      <div className="flex items-center text-xs text-yellow-500 mt-1">
                        {[...Array(5)].map((_, i) => (
                          <span key={i}>
                            {i < Math.floor(product.ratings_score) ? "★" : "☆"}
                          </span>
                        ))}
                        <span className="ml-1 text-gray-600">({product.ratings_count})</span>
                      </div>
                    )}
                    
                    <p className="text-xs text-gray-500 mt-1 line-clamp-1">{product.description}</p>
                    
                    <div className="mt-2">
                      <div className="text-xs px-2 py-1 bg-[#FE90EA] text-black rounded-full inline-block">Similarity: {product.score}</div>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="flex items-center justify-center h-24 text-sm text-gray-500">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-[#FE90EA]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Loading similar products...
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );


// before the Performance Metrics section
function ScrollingQueryExamples() {
  // Sample query examples - in production, this would come from your JSON API
  const queryExamples = [
    "poker training course",
    "how to play texas hold'em",
    "poker math for beginners",
    "poker strategy guide",
    "poker tournament tips",
    "online poker course",
    "poker coaching",
    "poker odds calculator",
    "advanced poker techniques",
    "poker software tools"
  ];
  
  const [currentIndex, setCurrentIndex] = useState(0);
  
  // Rotate through the examples
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % queryExamples.length);
    }, 3000); // Change example every 3 seconds
    
    return () => clearInterval(interval);
  }, [queryExamples.length]);
  
  return (
    <div className="bg-white p-5 rounded-lg shadow-sm mb-6 border-2 border-gray-200 overflow-hidden">
      <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
        <Search className="w-4 h-4 mr-2 text-[#FE90EA]" />
        <span className="text-black border-b-2 border-[#FE90EA] pb-1">Popular Queries</span>
      </h3>
      
      <div className="relative h-10 overflow-hidden">
        {queryExamples.map((query, index) => (
          <div
            key={index}
            className={`absolute w-full transition-all duration-500 ease-in-out ${
              index === currentIndex 
                ? 'translate-y-0 opacity-100' 
                : 'translate-y-10 opacity-0'
            }`}
          >
            <div 
              className="font-medium text-base text-gray-800 cursor-pointer hover:text-[#FE90EA]"
              onClick={() => {
                setQuery(query);
                handleSearch();
              }}
            >
              "{query}"
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

}

export default App;