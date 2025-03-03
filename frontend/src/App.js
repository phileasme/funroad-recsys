import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Search, BarChart as BarChartIcon, PieChart, Layers, Settings, TrendingUp, Sun, Moon, TrendingUpDown } from 'lucide-react';
import { searchProducts, getSimilarProducts } from './services/api';


// Import existing styles
const darkModeStyles = `
  /* Dark mode styles */
  body.dark-mode {
    background-color: #1a202c;
    color: #e2e8f0;
  }
  
  /* Adjust scrollbars for dark mode */
  body.dark-mode ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  
  body.dark-mode ::-webkit-scrollbar-track {
    background: #2d3748;
  }
  
  body.dark-mode ::-webkit-scrollbar-thumb {
    background-color: #4a5568;
    border-radius: 4px;
  }
  
  body.dark-mode ::-webkit-scrollbar-thumb:hover {
    background-color: #718096;
  }
  
  /* Adjust select dropdown for dark mode */
  body.dark-mode select option {
    background-color: #2d3748;
    color: #e2e8f0;
  }
`;

const customScrollbarStyles = `
  /* Custom scrollbar styles */
  .custom-scrollbar {
    scrollbar-width: thin;
    scrollbar-color: rgba(254, 144, 234, 0.5) transparent;
  }
  
  .custom-scrollbar::-webkit-scrollbar {
    width: 4px;
  }
  
  .custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
    margin: 3px 0;
  }
  
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background-color: rgba(254, 144, 234, 0.5);
    border-radius: 20px;
    border: 1px solid transparent;
  }
  
  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background-color: rgba(254, 144, 234, 0.8);
  }
  
  /* Dark mode specific scrollbar adjustments */
  body.dark-mode .custom-scrollbar::-webkit-scrollbar-thumb {
    background-color: rgba(254, 144, 234, 0.4);
  }
  
  body.dark-mode .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background-color: rgba(254, 144, 234, 0.7);
  }
  
  /* Hide scrollbar when not hovering (optional) */
  .custom-scrollbar.hide-scrollbar::-webkit-scrollbar-thumb {
    background-color: transparent;
  }
  
  .custom-scrollbar.hide-scrollbar:hover::-webkit-scrollbar-thumb {
    background-color: rgba(254, 144, 234, 0.5);
  }
  
  body.dark-mode .custom-scrollbar.hide-scrollbar:hover::-webkit-scrollbar-thumb {
    background-color: rgba(254, 144, 234, 0.4);
  }
`;



// Combine all styles
const allStyles = `
  ${darkModeStyles}
  ${customScrollbarStyles}
`;

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
  { id: 'search_fuzzy', name: 'Fuzzy Search' },
  { id: 'search_vision', name: 'Vision Search' },
  { id: 'search_colbert', name: 'Sentence Embedding Search' },
  { id: 'search_combined_v0_7', name: ' Combined No rating', version: "(v0.7)" },
  {id: 'search_combined_v0_8', name: 'Combine with ratings', version: "(v0.8)" },
  // 
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

function ProductCard({ product, index, darkMode, onHover, onLeave }) {
  // State to track hover
  const [isHovered, setIsHovered] = useState(false);
  // State to control actual display of details (with delayed timing)
  const [showDetails, setShowDetails] = useState(true);
  // State to control fullscreen image view
  const [showFullImage, setShowFullImage] = useState(false);
  // State to track if mouse is over image specifically
  const [mouseOverImage, setMouseOverImage] = useState(false);
  // State to track image aspect ratio
  const [isWideImage, setIsWideImage] = useState(false);
  
  // Refs for height calculation and timeout management
  const cardRef = useRef(null);
  const timeoutRef = useRef(null);
  const imageRef = useRef(null);
  
  // Store the original height of the card
  const [cardHeight, setCardHeight] = useState(null);
  
  const handleMouseEnter = (e) => {
    // Call the parent's hover handler
    if (onHover) onHover(product, e);
    
    // Set our local hover state
    setIsHovered(true);
  };
    
  const handleMouseLeave = () => {
    // Call the parent's leave handler
    if (onLeave) onLeave();
    
    // Reset our local hover state
    setIsHovered(false);
    setMouseOverImage(false);
  };
    
  // Check if product contains design-related keywords
  const isDesignRelated = useMemo(() => {
    const keywords = ['design', 'image', 'logo', 'picture', 'stitch'];
    const searchText = `${product.name} ${product.description}`.toLowerCase();
    return keywords.some(keyword => searchText.includes(keyword));
  }, [product.name, product.description]);
  
  // If it's design related and NOT a wide image, we want to show it in "hover mode" by default
  const shouldShowAsHover = (isHovered || (isDesignRelated && !isWideImage));
  
  // Effect to calculate and store card height on mount
  useEffect(() => {
    if (cardRef.current) {
      // Use a fixed height for cards that better matches the reference images
      const height = 400; // Adjusted to match reference
      setCardHeight(height);
    }
  }, []);
  
  // Effect to handle image load and check aspect ratio
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const isWide = img.width > img.height;
      setIsWideImage(isWide);
    };
    img.src = product.thumbnail_url || `https://placehold.co/600x400?text=${encodeURIComponent(product.name)}`;
    
    return () => {
      img.onload = null; // Clean up
    };
  }, [product.thumbnail_url, product.name]);
  
  // Effect to handle delayed state changes
  useEffect(() => {
    // Clear any existing timeouts
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    if (shouldShowAsHover) {
      // Immediately hide details when hovering starts
      setShowDetails(false);
    } else {
      // When hover ends, wait for transition to complete before showing details
      timeoutRef.current = setTimeout(() => {
        setShowDetails(true);
      }, 50); // Small delay to ensure transitions complete
    }
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [shouldShowAsHover]);
  
  const handleImageMouseEnter = () => {
    setMouseOverImage(true);
  };
  
  const handleImageMouseLeave = () => {
    setMouseOverImage(false);
  };
  
  const handleImageClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setShowFullImage(true);
  };
  
  const handleCloseFullImage = () => {
    setShowFullImage(false);
  };
  
  return (
    <>
      <div
        ref={cardRef}
        className={`${darkMode ? 'bg-gray-700 border-gray-600 hover:border-[#FE90EA]' : 'bg-white border-gray-200 hover:border-[#FE90EA]'} border-2 rounded-lg overflow-hidden hover:shadow-lg transition-shadow product-card`}
        style={{ 
          height: cardHeight ? `${cardHeight}px` : 'auto', 
          position: 'relative',
          maxHeight: '300px' // Adjusted to match reference
        }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <a href={product.url || "#"} target="#" onClick={(e) => mouseOverImage && shouldShowAsHover && e.preventDefault()}>
          {/* Image container - always present */}
          <div 
            style={{ 
              position: shouldShowAsHover ? 'absolute' : 'relative',
              top: 0,
              left: 0,
              width: '100%',
              height: shouldShowAsHover ? '100%' : '160px',
              padding: shouldShowAsHover ? '8px' : '0',
              transition: 'padding 0.3s ease, height 0.3s ease',
              zIndex: 1,
              display: 'flex',
              alignItems: shouldShowAsHover ? 'flex-start' : 'center',
              justifyContent: 'center',
              cursor: shouldShowAsHover && mouseOverImage ? 'zoom-in' : 'pointer',
            }}
            onMouseEnter={handleImageMouseEnter}
            onMouseLeave={handleImageMouseLeave}
            onClick={shouldShowAsHover && mouseOverImage ? handleImageClick : undefined}
          >
            <img 
              ref={imageRef}
              src={product.thumbnail_url || `https://placehold.co/600x400?text=${encodeURIComponent(product.name)}`} 
              alt={product.name} 
              style={{
                width: '100%',
                height: '100%',
                objectFit: shouldShowAsHover ? 'contain' : 'cover',
                objectPosition: shouldShowAsHover ? 'top' : 'center',
                transition: 'object-fit 0.3s ease, object-position 0.3s ease',
              }}
              onError={(e) => {
                e.target.src = `https://placehold.co/600x400?text=${encodeURIComponent(product.name.substring(0, 20))}`;
              }}
            />
            
            {/* Magnifier icon - only visible when mouse is over the image during hover */}
            {shouldShowAsHover && mouseOverImage && (
              <div 
                style={{
                  position: 'absolute',
                  top: '16px',
                  right: '16px',
                  backgroundColor: darkMode ? 'rgba(0, 0, 0, 0.5)' : 'rgba(255, 255, 255, 0.5)',
                  borderRadius: '50%',
                  width: '32px',
                  height: '32px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backdropFilter: 'blur(2px)',
                  boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
                }}
              >
                <svg 
                  width="20"
                  height="20"
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke={darkMode ? "white" : "black"} 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                >
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                  <line x1="11" y1="8" x2="11" y2="14"></line>
                  <line x1="8" y1="11" x2="14" y2="11"></line>
                </svg>
              </div>
            )}
          </div>
          
          {/* Overlay description - only visible when in hover mode */}
          {shouldShowAsHover && (
            <div
              style={{
                position: 'absolute',
                bottom: '8px',
                left: '8px',
                right: '8px',
                zIndex: 20,
                backgroundColor: darkMode ? 'rgba(26, 32, 44, 0.85)' : 'rgba(255, 255, 255, 0.85)',
                padding: '8px',
                borderRadius: '6px',
                backdropFilter: 'blur(2px)',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
              }}
            >
              <h3 className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-800'} mb-1 line-clamp-1`}>
                {product.name}
              </h3>
              <div className="flex items-center mb-1">
                {product.ratings_score !== undefined && (
                  <>
                    <div className="flex text-yellow-400">
                      {[...Array(5)].map((_, i) => (
                        <span key={i} className="text-xs">
                          {i < Math.floor(product.ratings_score) ? "★" : "☆"}
                        </span>
                      ))}
                    </div>
                    <span className={`ml-1 text-xs ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      {product.ratings_score} {product.ratings_count > 0 ? `(${product.ratings_count})` : ''}
                    </span>
                  </>
                )}
              </div>
              <div className="flex items-center justify-between mt-1">
                <div className="flex items-center">
                  <span className={`text-xs font-medium ${darkMode ? 'text-gray-200' : 'text-gray-700'} mr-2`}>
                    ${(product.price_cents / 100).toFixed(2)}
                  </span>
                  <span className={`inline-flex items-center px-1 py-0.5 rounded-md ${darkMode ? 'bg-gray-700 text-gray-200' : 'bg-gray-200 text-black'} text-xs font-medium`}>
                    Score: {typeof product.score === 'number' ? product.score.toFixed(2) : product.score}
                  </span>
                </div>
                
                <a 
                  href={product.url || "#"} 
                  target="#"
                  className="inline-flex items-center justify-center px-2 py-1 text-xs font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-1 focus:ring-[#FE90EA] border border-black"
                  onClick={(e) => e.stopPropagation()}
                >
                  View details
                </a>
              </div>
            </div>
          )}
          
          {/* Price tag - always visible when not in hover mode */}
          {product.price_cents !== undefined && !shouldShowAsHover && (
            <div className="absolute rounded-md top-3 right-3 flex items-center" style={{ zIndex: 30 }}>
              <div className="relative rounded-md bg-[#FE90EA] text-black font-medium py-0 px-1 text-base border border-t-transparent border-l-black border-r-transparent border-b-black">
                ${(product.price_cents / 100).toFixed(2)}
                <div className="absolute -right-[4px] -top-[1px] w-0 h-0 border-t-[8px] border-b-[7px] border-l-[5px] border-t-transparent border-b-transparent border-l-black"></div>
                <div className="absolute -right-[4px] bottom-[1px] w-0 h-0 border-t-[7px] border-b-[7px] border-l-[5px] border-t-transparent border-b-transparent border-l-[#FE90EA]"></div>
              </div>
            </div>
          )}
          
          {/* Design related label - only visible for design products when not hovered */}
          {isDesignRelated && !isHovered && (
            <div 
              className={`absolute top-2 left-2 text-xs font-medium px-1.5 py-0.5 rounded-full ${
                darkMode ? 'bg-blue-600 text-white' : 'bg-blue-100 text-blue-800'
              }`}
              style={{ zIndex: 25 }}
            >
              Design
            </div>
          )}
          
          {/* Standard details section - only rendered when not in hover mode */}
          {showDetails && !shouldShowAsHover && (
            <div 
              style={{ 
                padding: '0.75rem',
              }}
            >
              <h3 className={`font-medium text-sm ${darkMode ? 'text-gray-100' : 'text-gray-800'} mb-1 line-clamp-1`}>{product.name}</h3>
              
            
              {/* Rating display with stars */}
              { product.ratings_count > 0 && product.ratings_score !== undefined && (
                <div className="flex items-center mb-1">
                  <div className="flex text-yellow-400">
                    {[...Array(5)].map((_, i) => (
                      <span key={i} className="text-xs">
                        {i < Math.floor(product.ratings_score) ? "★" : "☆"}
                      </span>
                    ))}
                  </div>
                  <span className={`ml-1 text-xs ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    {product.ratings_score} {product.ratings_count > 0 ? `(${product.ratings_count})` : ''}
                  </span>
                </div>
              )}
              
              <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} text-xs mb-2 line-clamp-2`}>
                {product.description || "No description available."}
              </p>
              
              <div className="flex items-center justify-between mt-auto pt-1 border-t border-gray-100">
                <span className={`inline-flex items-center px-1.5 py-0.5 rounded-md ${darkMode ? 'bg-gray-600 text-gray-200' : 'bg-black/5 text-black'} text-xs font-medium`}>
                  Score: {typeof product.score === 'number' ? product.score.toFixed(2) : product.score}
                </span>
                
                <a 
                  href={product.url || "#"} 
                  target="#"
                  className="inline-flex items-center justify-center px-2 py-1 text-xs font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-1 focus:ring-[#FE90EA] border border-black"
                >
                  View details
                </a>
              </div>
            </div>
          )}
        </a>
      </div>
      
      {showFullImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-90"
          onClick={handleCloseFullImage}
        >
          <div 
            className="relative max-w-4xl max-h-[85vh] w-full h-full flex items-center justify-center"
            onClick={(e) => e.stopPropagation()}
          >
            <button 
              className="absolute top-4 right-4 z-10 w-8 h-8 rounded-full bg-black bg-opacity-50 flex items-center justify-center text-white hover:bg-opacity-70 transition-colors"
              onClick={handleCloseFullImage}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
            
            <img 
              src={product.thumbnail_url || `https://placehold.co/1200x800?text=${encodeURIComponent(product.name)}`} 
              alt={product.name}
              className="max-w-full max-h-full object-contain cursor-pointer"
              onClick={() => window.open(product.url || "#", "_blank")}
              onError={(e) => {
                e.target.src = `https://placehold.co/1200x800?text=${encodeURIComponent(product.name.substring(0, 20))}`;
              }}
            />
                  
            <div className="absolute bottom-4 left-0 right-0 text-center text-white bg-black bg-opacity-50 py-2 px-4">
              <h3 className="font-bold text-base">{product.name}</h3>
              <p className="text-sm opacity-90">${(product.price_cents / 100).toFixed(2)}</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function App() {


  const [previewProduct, setPreviewProduct] = useState(null);
  const [query, setQuery] = useState('poker');
  const [searchProfile, setSearchProfile] = useState('search_combined_v0_8');
  const [searchResults, setSearchResults] = useState(false);
  const [similarProducts, setSimilarProducts] = useState([]);
  const [hoveredProduct, setHoveredProduct] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [performanceData, setPerformanceData] = useState(defaultProfileData.search_combined_simplified_but_slow);
  const [searchHistory, setSearchHistory] = useState([
    { query: 'poker', timestamp: new Date().toLocaleTimeString(), results: 6, queryTime: 145 }
  ]);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [darkMode, setDarkMode] = useState(true);
  const [showSimilarProducts, setShowSimilarProducts] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [isFirstSearch, setIsFirstSearch] = useState(true);
  const [showLoadingSpinner, setShowLoadingSpinner] = useState(false);
  const loadingTimerRef = useRef(null);
  const similarProductsRef = useRef(null);
  const searchInputRef = useRef(null);

// Make sure the useEffect for style injection is in place

useEffect(() => {
  // Create style element
  const styleElement = document.createElement('style');
  styleElement.innerHTML = allStyles;
  document.head.appendChild(styleElement);
  
  // Cleanup when component unmounts
  return () => {
    document.head.removeChild(styleElement);
  };
}, []);


// Create handlers for mouse enter and leave
const handleProductPreviewEnter = (productId) => {
  setPreviewProduct(productId);
};

const handleProductPreviewLeave = () => {
  setPreviewProduct(null);
};




  function RecentSearchesComponent({ searchHistory, setQuery, handleSearch, darkMode }) {
    if (!searchHistory || searchHistory.length === 0) return null;
    
    return (
      <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} p-5 rounded-lg shadow-sm mb-6 border-2`}>
        <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-3 flex items-center`}>
          <Search className="w-4 h-4 mr-2 text-[#FE90EA]" />
          <span className={`${darkMode ? 'text-white' : 'text-black'} border-b-2 border-[#FE90EA] pb-1`}>Your Recent Searches</span>
        </h3>
        
        <div className="space-y-2 max-h-36 overflow-y-auto pr-1">
          {searchHistory.map((item, index) => (
            <div 
              key={index}
              className={`${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-50 hover:bg-gray-100'} px-3 py-2 rounded-md cursor-pointer transition-colors flex justify-between items-center`}
              onClick={() => { setQuery(item.query); handleSearch(); }}
            >
              <div className="flex items-center">
                <Search className="w-3 h-3 text-[#FE90EA] mr-2" />
                <span className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                  "{item.query}"
                </span>
              </div>
              <div className="flex items-center">
                <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'} mr-2`}>
                  {item.results} results
                </span>
                <span className="text-xs text-[#FE90EA]">
                  {item.queryTime.toFixed(2)}ms
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  // Charts and metrics section completion
function PerformanceCharts({ performanceData, darkMode }) {
  return (
    <>
      {/* Comparison chart */}
      <div className="mt-6">
        <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>Metrics Comparison</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={performanceData.comparisonChart}
              barSize={15}
              layout="vertical"
            >
              <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#555" : "#ccc"} />
              <XAxis 
                type="number" 
                domain={[0, 1]} 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} 
                stroke={darkMode ? "#aaa" : "#666"}
              />
              <YAxis 
                type="category" 
                dataKey="name" 
                width={70} 
                stroke={darkMode ? "#aaa" : "#666"}
              />
              <Tooltip 
                formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Score']}
                contentStyle={{ 
                  backgroundColor: darkMode ? '#2d3748' : '#fff',
                  borderColor: darkMode ? '#4a5568' : '#e2e8f0',
                  color: darkMode ? '#e2e8f0' : '#1a202c'
                }}
              />
              <Legend />
              <Bar dataKey="current" fill="#3B82F6" name="Current" />
              <Bar dataKey="baseline" fill={darkMode ? "#6B7280" : "#9CA3AF"} name="Baseline" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Time series chart */}
      <div className="mt-6">
        <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>Response Time Trend (ms)</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={performanceData.timeData}>
              <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#555" : "#ccc"} />
              <XAxis 
                dataKey="name" 
                stroke={darkMode ? "#aaa" : "#666"}
              />
              <YAxis 
                stroke={darkMode ? "#aaa" : "#666"}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: darkMode ? '#2d3748' : '#fff',
                  borderColor: darkMode ? '#4a5568' : '#e2e8f0',
                  color: darkMode ? '#e2e8f0' : '#1a202c'
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="current" stroke="#3B82F6" name="Current" strokeWidth={2} />
              <Line type="monotone" dataKey="baseline" stroke={darkMode ? "#6B7280" : "#9CA3AF"} name="Baseline" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}


function LoadingSpinner({ query, darkMode }) {
  return (
    <div className={`flex flex-col items-center justify-center ${darkMode ? 'bg-gray-800' : 'bg-white'} p-12 rounded-lg shadow-sm text-center`}>
      <svg 
        className="animate-spin h-16 w-16 text-[#FE90EA] mb-4" 
        xmlns="http://www.w3.org/2000/svg" 
        fill="none" 
        viewBox="0 0 24 24"
      >
        <circle 
          className="opacity-25" 
          cx="12" 
          cy="12" 
          r="10" 
          stroke="currentColor" 
          strokeWidth="4"
        ></circle>
        <path 
          className="opacity-75" 
          fill="currentColor" 
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        ></path>
      </svg>
      <p className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        Searching for {query} products...
      </p>
      <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'} mt-2`}>
        Getting the best results for you
      </p>
    </div>
  );
}



  useEffect(() => {
    performSearch('poker');
  }, []);  

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    performSearch(query);
  };
  

  const performSearch = async (searchQuery) => {
    if (!searchQuery.trim()) return;
    
    // Set loading state
    setIsLoading(true);
    
    // Clear any existing timers first
    if (loadingTimerRef.current) {
      clearTimeout(loadingTimerRef.current);
      loadingTimerRef.current = null;
    }
    
    // Set spinner state based on whether it's first search
    if (isFirstSearch) {
      setShowLoadingSpinner(true);
      setIsFirstSearch(false);
    } else {
      // Initially hide spinner for non-first searches
      setShowLoadingSpinner(false);
      
      // Set timer to show spinner after 600ms
      loadingTimerRef.current = setTimeout(() => {
        setShowLoadingSpinner(true);
      }, 600);
    }
    
    try {
      // API call and search logic
      const data = await searchProducts(searchProfile, searchQuery);
      setSearchResults(data.results || []);
      
      // Update search history
      setSearchHistory(prev => [
        { query: searchQuery, timestamp: new Date().toLocaleTimeString(), results: data.results?.length || 0, queryTime: data.query_time_ms },
        ...prev.slice(0, 9)
      ]);
    } catch (error) {
      console.error('Search error:', error);
      // Fallback to mock data after a delay to simulate network request
      setTimeout(() => {
        setSearchResults(Array(6).fill(0).map((_, i) => ({
          id: `result-${i}`,
          score: (Math.random() * 0.2 + 0.8).toFixed(2),
          name: `${searchQuery.charAt(0).toUpperCase() + searchQuery.slice(1)} Product ${i + 1}`,
          description: `This is a sample product related to "${searchQuery}".`,
          thumbnail_url: `https://placehold.co/600x400?text=${encodeURIComponent(searchQuery)}+${i+1}`,
          price_cents: Math.floor(Math.random() * 5000) + 1000,
          ratings_score: (Math.random() * 1 + 4).toFixed(1),
          ratings_count: Math.floor(Math.random() * 300) + 50,
          url: '#'
        })));
      }, 1000);
    } finally {
      // IMPORTANT: Clear the spinner timer immediately
      if (loadingTimerRef.current) {
        clearTimeout(loadingTimerRef.current);
        loadingTimerRef.current = null;
      }
      
      // If search completed before spinner showed, don't show it at all
      setShowLoadingSpinner(false);
      setIsLoading(false);
    }
  };
  
  // Make sure to clean up on unmount
  useEffect(() => {
    return () => {
      if (loadingTimerRef.current) {
        clearTimeout(loadingTimerRef.current);
      }
    };
  }, []);
// Add these new refs
const hoverTimerRef = useRef(null);
const similarProductsScrollRef = useRef(null);
const [isProductHovered, setIsProductHovered] = useState(false);
const [isPopupHovered, setIsPopupHovered] = useState(false);
const hoverIntentTimerRef = useRef(null);
const blindSpotTimerRef = useRef(null);

const handleProductHover = useCallback(async (product, event) => {
  // Clear any existing hover timers
  if (hoverTimerRef.current) {
    clearTimeout(hoverTimerRef.current);
  }
  if (blindSpotTimerRef.current) {
    clearTimeout(blindSpotTimerRef.current);
  }
  
  // Set product as being hovered
  setIsProductHovered(true);
  setHoveredProduct(product);
  
  // Get the dimensions and position of the product card
  const productCard = event.currentTarget;
  const rect = productCard.getBoundingClientRect();
  
  // Store hover position
  setHoverPosition({ 
    x: rect.right + 10,
    y: rect.top 
  });
  
  // Start fetching the data immediately
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
      name: `Similar (images) to ${product.name} - Item ${i + 1}`,
      description: `A product similar to ${product.name}.`,
      thumbnail_url: `https://placehold.co/50x50?text=Similar+${i+1}`,
      score: (Math.random() * 0.3 + 0.7).toFixed(2),
      ratings_count: Math.floor(Math.random() * 100) + 5,
      ratings_score: (Math.random() * 1 + 4).toFixed(1),
      price_cents: Math.floor(Math.random() * 5000) + 500,
      url: '#'
    }));
    
    setSimilarProducts(fakeSimilarProducts);
  }
  
  // // Set timer to show the similar products after a short delay
  hoverTimerRef.current = setTimeout(() => {
    setSelectedProduct(product);
    setShowSimilarProducts(true);
    
    // Reset scroll position when showing popup
    if (similarProductsScrollRef.current) {
      similarProductsScrollRef.current.scrollTop = 0;
    }
  }, 5);
}, []);

const closeSimilarProducts = useCallback(() => {
  setShowSimilarProducts(false);
  setHoveredProduct(null);
  setSelectedProduct(null);
  setSimilarProducts([]);
  
  // Clear any pending timers
  if (hoverTimerRef.current) {
    clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = null;
  }
  if (blindSpotTimerRef.current) {
    clearTimeout(blindSpotTimerRef.current);
    blindSpotTimerRef.current = null;
  }
}, []);

const handleProductMouseLeave = useCallback(() => {
  setIsProductHovered(false);
  
  // Use a delay before closing to handle the blind spot
  blindSpotTimerRef.current = setTimeout(() => {
    // Only close if popup is not being hovered
    if (!isPopupHovered) {
      closeSimilarProducts();
    }
  }, 100); 
}, [isPopupHovered, closeSimilarProducts]);

const handlePopupMouseLeave = useCallback(() => {
  setIsPopupHovered(false);
  
  // Close after a short delay if product is not hovered
  blindSpotTimerRef.current = setTimeout(() => {
    if (!isProductHovered) {
      closeSimilarProducts();
    }
  }, 100);
}, [isProductHovered, closeSimilarProducts]);

// Handle similar products popup mouse enter
const handlePopupMouseEnter = () => {
  // Set popup as being hovered
  setIsPopupHovered(true);
  
  // Clear any pending blind spot timer
  if (blindSpotTimerRef.current) {
    clearTimeout(blindSpotTimerRef.current);
    blindSpotTimerRef.current = null;
  }
};


// Clean up timers on unmount
useEffect(() => {
  return () => {
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
    }
    if (blindSpotTimerRef.current) {
      clearTimeout(blindSpotTimerRef.current);
    }
    if (hoverIntentTimerRef.current) {
      clearTimeout(hoverIntentTimerRef.current);
    }
  };
}, []);

useEffect(() => {
  const handleDocumentClick = (e) => {
    if (showSimilarProducts) {
      const popupElement = similarProductsRef.current;
      
      if (popupElement && !popupElement.contains(e.target)) {
        // Check if click was on a product card
        const isProductCard = e.target.closest('.product-card');
        if (!isProductCard) {
          closeSimilarProducts();
        }
      }
    }
  };
  
  document.addEventListener('click', handleDocumentClick);
  return () => document.removeEventListener('click', handleDocumentClick);
}, [showSimilarProducts, closeSimilarProducts]);


  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  // Apply dark mode class to body
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  }, [darkMode]);

  return (
    <div className={`flex flex-col min-h-screen ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-black'}`}>
      {/* Header */}
      <header className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm py-4 px-6 border-b-2 border-[#FE90EA]`}>
          <div className="max-w-7xl mx-auto flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <img src="/gumroad.png" alt="Gumroad Logo" className="h-8 w-auto" />
              <h1 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-black'}`}>Gumroad Search Prototype</h1>
            </div>
            
            {/* Middle section with nav links */}
            <div className="hidden md:flex items-center space-x-6">
              <div>
              <a 
                href="https://www.notion.so/Search-Discovery-Case-Study-Blog-40e476a45ad94596ad323289eac62c2c" 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center px-3 py-1 text-xs font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-1 focus:ring-[#FE90EA] border border-black"
              >
                Case Study
              </a>
              </div>
              <div>  
              <a 
                href="https://phileas.me" 
                target="_blank" 
                rel="noopener noreferrer"
                className={`text-sm ${darkMode ? 'text-gray-300 hover:text-white' : 'text-gray-700 hover:text-black'} transition-colors flex items-center`}
              >
                <span className="mr-1">By</span>
                <span className="font-medium text-[#FE90EA]">Phileas Hocquard</span>
              </a>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-black'}`}>
                Indexed Products: 5,467
              </div>
              <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-black'}`}>
                Total Shards: 1 (be gentle 🙏)
              </div>
              <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-black'}`}>
                Search Profile: v0.8
              </div>
              {/* Dark mode toggle */}
              <button 
                onClick={toggleDarkMode} 
                className={`p-2 rounded-full ${darkMode ? 'bg-gray-700 text-yellow-400' : 'bg-gray-200 text-gray-700'}`}
                aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </button>
              {/* <Settings className="h-5 w-5 text-[#FE90EA] cursor-pointer hover:text-gray-400" /> */}
            </div>
          </div>
          
          {/* Mobile navigation - only shown on small screens */}
          <div className="md:hidden mt-2 pt-2 border-t border-gray-700 flex justify-center space-x-6">
            <a 
              href="https://www.notion.so/Search-Discovery-Case-Study-Blog-40e476a45ad94596ad323289eac62c2c" 
              target="_blank" 
              rel="noopener noreferrer"
              className={`text-xs ${darkMode ? 'text-gray-300 hover:text-white' : 'text-gray-700 hover:text-black'} transition-colors`}
            >
              Case Study
            </a>
            <div className="h-4 border-r border-gray-400"></div>
            <a 
              href="https://phileas.me" 
              target="_blank" 
              rel="noopener noreferrer"
              className={`text-xs ${darkMode ? 'text-gray-300 hover:text-white' : 'text-gray-700 hover:text-black'} transition-colors`}
            >
              By Phileas Hocquard
            </a>
          </div>
        </header>
      {/* Main content */}
      <main className="flex-grow py-6 px-6">
        <div className="w-full mx-auto mx-auto">
          {/* Search form - Common to both layouts */}
          <div className="flex justify-center w-full">
            {/* Search form - Common to both layouts */}
            <div className="flex justify-center w-full">
              <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} p-6 rounded-lg shadow-sm mb-6 border-2 w-full max-w-7xl mx-auto`}>
                <form onSubmit={handleSearch} className="flex items-center gap-4">
                  <div className="relative flex-grow">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-[#FE90EA]" />
                    <input
                      ref={searchInputRef}
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Search for products..."
                      className={`w-full pl-10 pr-4 py-3 rounded-md border-2 ${
                        darkMode 
                          ? 'border-gray-600 bg-gray-700 text-white focus:border-[#FE90EA]' 
                          : 'border-gray-300 bg-white text-black focus:border-[#FE90EA]'
                      } focus:outline-none focus:ring-1 focus:ring-[#FE90EA]`}
                      onClick={(e) => e.target.select()}
                    />
                  </div>
                  <select
                    value={searchProfile}
                    onChange={(e) => setSearchProfile(e.target.value)}
                    className={`px-3 py-3 rounded-md border-2 ${
                      darkMode 
                        ? 'border-gray-600 bg-gray-700 text-white' 
                        : 'border-gray-300 bg-white text-black'
                    } focus:outline-none focus:border-[#FE90EA] focus:ring-1 focus:ring-[#FE90EA] flex-shrink-0`}
                  >
                    {searchProfiles.map(profile => (
                      <option key={profile.id} value={profile.id}>
                        {profile.name} {profile.version && <span className="text-[#FE90EA]"> {profile.version}</span>}
                      </option>
                    ))}
                  </select>
                  <button
                    type="submit"
                    className="bg-[#FE90EA] text-black px-6 py-3 rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-2 focus:ring-[#FE90EA] focus:ring-offset-2 font-medium border-2 border-black flex-shrink-0"
                    disabled={isLoading}
                  >
                    {isLoading ? 'Searching...' : 'Search'}
                  </button>
                </form>
              </div>
            </div>
            </div>
          {/* Two-column layout for desktop, stacked for mobile */}
          <div className="flex flex-col lg:flex-row gap-6">
            {/* Left column (wider) - Search results */}
            <div className="lg:w-2/3 space-y-6">
              {/* Search results or loading state */}
              {isLoading && showLoadingSpinner ? (
                <LoadingSpinner darkMode={darkMode} query={query}/>
              ) : searchResults.length > 0 ? (
                <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-6 rounded-lg shadow-sm`}>
                  <h2 className={`text-xl font-semibold mb-6 ${darkMode ? 'text-white' : 'text-black'} border-b-2 border-[#FE90EA] pb-2 inline-block`}>
                    Search Results ({searchResults.length})
                  </h2>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                  {searchResults.map((product, index) => (
                    <ProductCard 
                      key={`${product.id || product.name}-${index}`}
                      product={product}
                      index={index}
                      darkMode={darkMode}
                      onHover={handleProductHover} // This was missing, should call handleProductHover
                      onLeave={handleProductMouseLeave} // This was missing, should call handleProductMouseLeave
                    />
                  ))}
                  </div>
                </div>
              ) : (
                <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-6 rounded-lg shadow-sm text-center`}>
                  <p className={`text-lg ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                    No results found for "{query}". Try a different search term.
                  </p>
                </div>
              )}

              {/* Search history section */}
              {searchHistory.length > 0 && (
                <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-6 rounded-lg shadow-sm`}>
                  <h2 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-black'}`}>Recent Searches</h2>
                  <div className="overflow-x-auto">
                    <table className={`min-w-full divide-y ${darkMode ? 'divide-gray-700' : 'divide-gray-200'}`}>
                      <thead className={darkMode ? 'bg-gray-700' : 'bg-gray-50'}>
                        <tr>
                          <th className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Query</th>
                          <th className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Time</th>
                          <th className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Results</th>
                          <th className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Query Time</th>
                          <th className={`px-6 py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Action</th>
                        </tr>
                      </thead>
                      <tbody className={`${darkMode ? 'bg-gray-800 divide-y divide-gray-700' : 'bg-white divide-y divide-gray-200'}`}>
                        {searchHistory.map((item, i) => (
                          <tr key={i} className={darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'}>
                            <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>{item.query}</td>
                            <td className={`px-6 py-4 whitespace-nowrap text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>{item.timestamp}</td>
                            <td className={`px-6 py-4 whitespace-nowrap text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>{item.results}</td>
                            <td className={`px-6 py-4 whitespace-nowrap text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>{item.queryTime?.toFixed(2) || '-'} ms</td>
                            <td className={`px-6 py-4 whitespace-nowrap text-sm text-blue-500 hover:text-blue-700`}>
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
            
            {/* Right column */}
            <div className="lg:w-1/3">
    
                {/* Scrolling Query Examples Component */}
                <ScrollingQueryExamples 
                setQuery={setQuery} 
                performSearch={performSearch}
                darkMode={darkMode}
              />



                <RecentSearchesComponent 
                  searchHistory={searchHistory.slice(0, 3)} 
                  setQuery={setQuery} 
                  handleSearch={handleSearch}
                  darkMode={darkMode}
                />

                
                <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-6 rounded-lg shadow-sm sticky top-6`}>
                  <h2 className={`text-lg font-semibold mb-4 flex items-center ${darkMode ? 'text-white' : 'text-black'}`}>
                    <TrendingUp className="mr-2 text-blue-600" />
                    Performance Metrics
                  </h2>
                  <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'} mb-4`}>
                    {searchProfiles.find(p => p.id === searchProfile)?.name}
                  </div>
                  
                  <div className="space-y-4">
                    {/* Metric cards */}
                    <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-50'} p-4 rounded-md`}>
                      <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'} mb-1`}>Precision</div>
                      <div className={`text-2xl font-bold ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>{performanceData.accuracy}%</div>
                      <div className="text-xs text-green-600 mt-1">+{(performanceData.accuracy - 62).toFixed(1)}% vs baseline</div>
                    </div>
                    
                    <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-50'} p-4 rounded-md`}>
                      <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'} mb-1`}>Recall</div>
                      <div className={`text-2xl font-bold ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>{performanceData.recall}%</div>
                      <div className="text-xs text-green-600 mt-1">+{(performanceData.recall - 65).toFixed(1)}% vs baseline</div>
                    </div>
                    
                    <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-50'} p-4 rounded-md`}>
                      <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'} mb-1`}>Avg. Latency</div>
                      <div className={`text-2xl font-bold ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>{performanceData.latency}ms</div>
                      <div className="text-xs text-red-600 mt-1">+{(performanceData.latency - 135).toFixed(1)}ms vs baseline</div>
                    </div>
                  </div>
                  
                  {/* Comparison chart */}
                  <div className="mt-6">
                    <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>Metrics Comparison</h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={performanceData.comparisonChart}
                          barSize={15}
                          layout="vertical"
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#555" : "#ccc"} />
                          <XAxis 
                            type="number" 
                            domain={[0, 1]} 
                            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} 
                            stroke={darkMode ? "#aaa" : "#666"}
                          />
                          <YAxis 
                            type="category" 
                            dataKey="name" 
                            width={70} 
                            stroke={darkMode ? "#aaa" : "#666"}
                          />
                          <Tooltip 
                            formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Score']}
                            contentStyle={{ 
                              backgroundColor: darkMode ? '#2d3748' : '#fff',
                              borderColor: darkMode ? '#4a5568' : '#e2e8f0',
                              color: darkMode ? '#e2e8f0' : '#1a202c'
                            }}
                          />
                          <Legend />
                          <Bar dataKey="current" fill="#3B82F6" name="Current" />
                          <Bar dataKey="baseline" fill={darkMode ? "#6B7280" : "#9CA3AF"} name="Baseline" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  
                  {/* Time series chart */}
                  <div className="mt-6">
                    <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>Response Time Trend (ms)</h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={performanceData.timeData}>
                          <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#555" : "#ccc"} />
                          <XAxis 
                            dataKey="name" 
                            stroke={darkMode ? "#aaa" : "#666"}
                          />
                          <YAxis 
                            stroke={darkMode ? "#aaa" : "#666"}
                          />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: darkMode ? '#2d3748' : '#fff',
                              borderColor: darkMode ? '#4a5568' : '#e2e8f0',
                              color: darkMode ? '#e2e8f0' : '#1a202c'
                            }}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="current" stroke="#3B82F6" name="Current" strokeWidth={2} />
                          <Line type="monotone" dataKey="baseline" stroke={darkMode ? "#6B7280" : "#9CA3AF"} name="Baseline" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  </div>
                </div>
              </div>
        </div> 

      </main>

          <div className="space-y-3">
            {/* Similar products popup */}
            {showSimilarProducts && selectedProduct && (
                  <div 
                    ref={(el) => {
                      similarProductsRef.current = el;
                      similarProductsScrollRef.current = el;
                    }}
                    className={`fixed ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-xl p-2 z-50 border-2 border-[#FE90EA] custom-scrollbar`}
                    style={{
                      top: `${90+Math.max(hoverPosition.y, 10)}px`,
                      left: `${hoverPosition.x}px`,
                      width: '300px', // Fixed width that matches reference
                      maxHeight: '320px', // Taller to match reference
                      overflowY: 'auto',
                    }}
                    onMouseEnter={handlePopupMouseEnter}
                    onMouseLeave={handlePopupMouseLeave}
                  >
                <div className="flex justify-between items-start pb-2">
                  <h4 className={`font-small text-sm flex-grow pr-2 border-b-2 border-[#FE90EA] ${darkMode ? 'text-white' : 'text-black'}`}>
                    Similarity (based on image) to:
                    <br/>
                    "{selectedProduct.name.substring(0, 25)}{selectedProduct.name.length > 25 ? '...' : ''}"
                  </h4>
                  
                  {/* Close Button */}
                  <button 
                    className={`p-1 rounded-full ${darkMode ? 'text-gray-300 hover:text-white hover:bg-gray-700' : 'text-gray-600 hover:text-black hover:bg-gray-100'}`}
                    onClick={closeSimilarProducts}
                    aria-label="Close similar products"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18"></line>
                      <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                  </button>
                </div>

                <div className="space-y-4">
                  {similarProducts.length > 0 ? (
                    similarProducts.map((product, index) => (
                      <a href={product.url || "#"} target="#" key={product.id || index} className="block">
                        <div 
                          className={`flex items-start py-2 ${darkMode ? 'border-gray-700 hover:bg-gray-700' : 'border-gray-100 hover:bg-gray-50'} border-b last:border-0 transition-colors rounded-md px-2`}
                        >
                          {/* Product Image */}
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
                          
                          {/* Product Details */}
                          <div className="ml-3 flex-grow min-w-0">
                            {/* Title and Price Row */}
                            <div className="flex justify-between items-start w-full">
                              <h4 className={`text-xs font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'} truncate max-w-[65%]`}>
                                {product.name}
                              </h4>
                              {product.price_cents !== undefined && (
                                <div className={`text-xs font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'} ml-1 flex-shrink-0`}>
                                  ${(product.price_cents / 100).toFixed(2)}
                                </div>
                              )}
                            </div>
                            
                            {/* Ratings - Only show if greater than 0 */}
                            {product.ratings_score > 0 && (
                              <div className="flex items-center text-xs text-yellow-500 mt-1">
                                {[...Array(5)].map((_, i) => (
                                  <span key={i} className="text-xs">
                                    {i < Math.floor(product.ratings_score) ? "★" : "☆"}
                                  </span>
                                ))}
                                <span className={`ml-1 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                  {product.ratings_count > 0 ? `(${product.ratings_count})` : ''}
                                </span>
                              </div>
                            )}
                            {/* Description */}
                            {product.description && (
                              <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'} mt-1 truncate`}>
                                {product.description}
                              </p>
                            )}
                            
                            {/* Similarity Score */}
                            <div className="mt-2">
                              <div className="text-xs px-2 py-0.5 bg-[#FE90EA] text-black rounded-full inline-block">
                                Similarity: {parseFloat(product.score).toFixed(2)}
                              </div>
                            </div>
                          </div>
                        </div>
                      </a>
                    ))
                  ) : (
                    <div className="flex items-center justify-center py-6 text-sm text-gray-500">
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
      
    <CopyrightFooter darkMode={darkMode} />
    </div>
  );

  function CopyrightFooter({ darkMode }) {
    const currentYear = new Date().getFullYear();
    
    return (
      <footer className={`mt-12 py-6 border-t ${darkMode ? 'border-gray-700 text-gray-400' : 'border-gray-200 text-gray-500'}`}>
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <div className="flex items-center">
                <span className="font-semibold text-sm mr-1">© {currentYear} Clusterise Inc.</span>
                <span className="text-xs">All Rights Reserved. </span>
                <span className="text-xs ml-2">
                Interface design and software © Clusterise Inc.
              </span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <a href="#" className={`text-xs hover:${darkMode ? 'text-white' : 'text-gray-800'} transition-colors`}>Terms</a>
              <a href="#" className={`text-xs hover:${darkMode ? 'text-white' : 'text-gray-800'} transition-colors`}>Privacy</a>
              <a href="#" className={`text-xs hover:${darkMode ? 'text-white' : 'text-gray-800'} transition-colors`}>Contact</a>
            </div>
          </div>
          
          <div className={`text-xs mt-4 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            This application is for demonstration purposes only. Gumroad is a trademark of Gumroad, Inc.
            Search interface is not affiliated with, endorsed by, or sponsored by Gumroad.
          </div>
        </div>
      </footer>
    );
  }
}


// Updated ScrollingQueryExamples Component
function ScrollingQueryExamples({ setQuery, performSearch, darkMode }) {
  // Sample query examples
  const queryExamples = ["ios 14 app icons", "kdp cover design", "python for beginners", "macbook air mockup", "ios 14 icons", "procreate brush pack", "mtt sng cash", "cross stitch pattern", "windows 11 themes", "max for live", "forex expert advisor", "figma ui kit", "kdp book cover", "cross stitch pdf", "ready to render", "macbook pro mockup", "ableton live packs", "kdp digital design", "royalty free music", "mt4 expert advisor", "sample pack", "betting system", "phone wallpaper", "design system", "tennis lessons", "poker online", "preset pack", "tennis course", "ai brushes", "lightroom bundle", "fishing logo", "instagram marketing", "oil painting", "notion template", "prompt engineering", "music production", "web design", "icon set", "abstract background", "pokertracker 4", "mobile mockup", "gambling tips", "sport car", "tennis training", "chatgpt mastery", "sports betting", "keyshot scene", "mockup template", "furry art", "football coach", "digital marketing", "lightroom preset", "amazon kdp", "ableton templates", "jersey 3d", "business marketing", "soccer drills", "macbook mockup", "business growth", "ui kit", "graphic design", "laptop mockup", "ios14 icons", "wallpaper phone", "vj clip", "design patterns", "john deere", "trading strategies", "vrchat avatar", "iphone mockup", "kdp interior", "free download", "ui design", "landing page", "vrchat accessories", "kids tennis", "wrapping papers", "apple mockup", "vj pack", "jersey template", "cheat sheet", "betfair trading", "fishing illustration", "wallpaper pack", "cross stitch", "motion graphics", "hand drawn", "diseño gráfico", "tennis technique", "notion layout", "vrchat asset", "ableton live", "poker tournaments", "zenbits gambling", "soccer training", "chatgpt course", "seamless clipart", "lightroom presets", "canva template", "tennis coaching", "sports trading", "best mom", "mobile app", "device mockup", "figma template", "iphone wallpaper", "digital art", "chatgpt tutorial", "3d model", "chatgpt prompts", "vrchat clothing", "business plan", "online poker", "hunting logo", "digital paper", "digital download", "procreate stamps", "notion templates", "digital painting", "clipart set", "lightroom mobile", "furry base", "tennis teaching", "jersey mockup", "icon pack", "after effects", "vector illustration", "poker ranges", "notion planner", "poker tool", "chatgpt resources", "procreate brush", "kdp book", "kdp template", "procreate brushes", "adobe illustrator", "design templates", "passive income", "dice control", "poker strategy", "social media", "vj loops", "notion dashboard", "subversive pattern", "betting models"];
  
  const [currentIndex, setCurrentIndex] = useState(0);
  const intervalRef = useRef(null);
  const lastUpdateTimeRef = useRef(Date.now());
  
  // Rotate through the examples with more resilient timing
  useEffect(() => {
    const updateIndex = () => {
      const now = Date.now();
      // Only update if 3 seconds have passed
      if (now - lastUpdateTimeRef.current >= 3000) {
        setCurrentIndex((prevIndex) => {
          let newIndex;
          do {
            newIndex = Math.floor(Math.random() * queryExamples.length);
          } while (newIndex === prevIndex); // Ensure it's different from the last index
          return newIndex;
        });
        lastUpdateTimeRef.current = now;
      }
      
      // Set up next animation frame
      intervalRef.current = requestAnimationFrame(updateIndex);
    };
    
    // Start the animation loop
    intervalRef.current = requestAnimationFrame(updateIndex);
    
    // Cleanup function
    return () => {
      if (intervalRef.current) {
        cancelAnimationFrame(intervalRef.current);
      }
    };
  }, [queryExamples.length]); // Only re-run if array length changes
  
  // Handle click on a query example
  const handleQueryClick = (query) => {
    setQuery(query);
    performSearch(query); // Make sure you're calling the passed function
  };
  
  return (
    <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} p-5 rounded-lg shadow-sm mb-6 border-2 overflow-hidden`}>
      <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2 flex items-center`}>
        <Search className="w-4 h-4 mr-2 text-[#FE90EA]" />
        <span className={`${darkMode ? 'text-white' : 'text-black'} border-b-2 border-[#FE90EA] pb-1`}>Popular Queries</span>
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
              className={`font-medium text-base ${darkMode ? 'text-gray-200 hover:text-[#FE90EA]' : 'text-gray-800 hover:text-[#FE90EA]'} cursor-pointer`}
              onClick={() => handleQueryClick(query)}
            >
              "{query}"
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;