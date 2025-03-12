
import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Search, BarChart as BarChartIcon, PieChart, Layers, Settings, TrendingUp, Sun, Moon, TrendingUpDown } from 'lucide-react';
import { searchProducts, getSimilarProducts } from './services/api';
import { processProductImages } from './services/imageService';
import SearchProfileSelector from './components/SearchProfileSelector';
import ProductCard from './components/ProductCard';
import { debounce } from 'lodash';
const SearchResultsWithSellerFilter = React.lazy(() => import('./components/SearchResultsWithSellerFilter'));


// Import existing styles - keeping this section as is
const AppStyles = `
.App {
  text-align: center;
}

.App-logo {
  height: 40vmin;
  pointer-events: none;
}

@media (prefers-reduced-motion: no-preference) {
  .App-logo {
    animation: App-logo-spin infinite 20s linear;
  }
}

.App-header {
  background-color: #282c34;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-size: calc(10px + 2vmin);
  color: white;
}

.App-link {
  color: #61dafb;
}

@keyframes App-logo-spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}
`;

// Dark mode styles - keeping this section as is
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

// Custom scrollbar styles - keeping this section as is
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

// Added responsive styles to improve mobile appearance
const responsiveStyles = `
  /* Responsive styles */
  @media (max-width: 768px) {
    .search-form {
      flex-direction: column;
      gap: 0.75rem;
    }
    
    .search-form input, 
    .search-form select, 
    .search-form button {
      width: 100%;
    }
    
    .metrics-card {
      margin-bottom: 1rem;
    }
    
    .table-responsive {
      display: block;
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    
    .history-table th,
    .history-table td {
      white-space: nowrap;
      padding: 0.5rem 0.75rem;
    }
  }
  
  @media (max-width: 640px) {
    .product-grid {
      grid-template-columns: repeat(1, 1fr) !important;
    }
    
    .metrics-section {
      padding: 1rem;
    }
    
    .chart-container {
      height: 200px !important;
    }
  }
  
  /* Fix search field on smaller screens */
  @media (max-width: 480px) {
    .main-header {
      padding: 0.75rem;
    }
    
    .main-header h1 {
      font-size: 1rem;
    }
    
    .page-content {
      padding: 0.75rem;
    }
  }
`;

// Combine all styles
const allStyles = `
  ${AppStyles}
  ${darkModeStyles}
  ${customScrollbarStyles}
  ${responsiveStyles}
`;

// Default profile data - keeping this section as is
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

const queryExamples = ["ios 14 app icons", "kdp cover design", "python for beginners", "macbook air mockup", "ios 14 icons", "procreate brush pack", "mtt sng cash", "cross stitch pattern", "windows 11 themes", "max for live", "forex expert advisor", "figma ui kit", "kdp book cover", "cross stitch pdf", "ready to render", "macbook pro mockup", "ableton live packs", "kdp digital design", "royalty free music", "mt4 expert advisor", "sample pack", "betting system", "phone wallpaper", "design system", "preset pack", "tennis course", "lightroom bundle", "fishing logo", "oil painting", "notion template", "prompt engineering", "music production", "web design", "icon set", "abstract background", "pokertracker 4", "mobile mockup", "gambling", "sport car", "tennis training", "chatgpt mastery", "sports betting", "keyshot scene", "mockup template", "furry art", "football coach", "lightroom preset", "amazon kdp", "jersey 3d", "business marketing", "macbook mockup", "business growth", "ui kit", "graphic design", "laptop mockup", "ios14 icons", "wallpaper phone", "vj clip", "design patterns", "john deere", "vrchat avatar", "iphone mockup", "kdp interior", "ui design", "landing page", "vrchat accessories", "kids tennis", "wrapping papers", "apple mockup", "vj pack", "cheat sheet", "betfair trading", "fishing illustration", "wallpaper pack", "cross stitch", "motion graphics", "hand drawn", "diseño gráfico", "tennis technique", "notion layout", "vrchat asset", "ableton live", "poker tournaments", "zenbits gambling", "soccer training", "chatgpt course", "seamless clipart", "lightroom presets", "canva template", "sports trading", "best mom", "device mockup", "figma template", "iphone wallpaper", "digital art", "chatgpt tutorial", "3d model", "chatgpt prompts", "vrchat clothing", "business plan", "online poker", "hunting logo", "digital paper", "digital download", "procreate stamps", "notion templates", "digital painting", "clipart set", "lightroom mobile", "furry base", "tennis teaching", "jersey mockup", "icon pack", "after effects", "vector illustration", "notion planner", "procreate brush", "kdp book", "kdp template", "procreate brushes", "adobe illustrator", "design templates", "passive income", "dice control", "poker strategy", "social media", "vj loops", "notion dashboard", "subversive pattern"];

const searchProfiles = [
  { id: 'search_fuzzy', name: 'Fuzzy Search' },
  { id: 'search_vision', name: 'Vision Search' },
  { id: 'search_colbert', name: 'Sentence Embedding Search' },
  {id: 'exact_match', name: 'ExactMatch'},
  { id: 'search_combined_v0_7', name: ' Combined No rating', version: "(v0.7)" },
  {id: 'search_combined_v0_8', name: 'Combine with ratings', version: "(v0.8)" },
  {id: 'two_phase_unnative_optimized', name: "2phase jit", version: "(v0.10)"},
  {id: 'elasticsearch_vector_optimized', name: "2phase es", version: "(v0.11)"},
  {id: 'two_phase_optimized', name: "2phase fallback", version: "(v0.12)"}

];

searchProfiles.reverse().forEach(profile => {
  if (!defaultProfileData[profile.id]) {
    defaultProfileData[profile.id] = {
      ...defaultProfileData.default,
      accuracy: defaultProfileData.default.accuracy + Math.floor(Math.random() * 15),
      recall: defaultProfileData.default.recall + Math.floor(Math.random() * 15)
    };
  }
});

function App() {
  const [previewProduct, setPreviewProduct] = useState(null);
  const [query, setQuery] = useState('');
  const [searchProfile, setSearchProfile] = useState('two_phase_optimized');
  const [searchResults, setSearchResults] = useState([]);
  const [similarProducts, setSimilarProducts] = useState([]);
  const [hoveredProduct, setHoveredProduct] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [performanceData, setPerformanceData] = useState(defaultProfileData.search_combined_simplified_but_slow);
  const [searchHistory, setSearchHistory] = useState([]);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [darkMode, setDarkMode] = useState(true);
  const [showSimilarProducts, setShowSimilarProducts] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [isFirstSearch, setIsFirstSearch] = useState(true);
  const [showLoadingSpinner, setShowLoadingSpinner] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [tabView, setTabView] = useState('results'); // 'results', 'history', 'metrics'
  
  const [selectedSeller, setSelectedSeller] = useState(null);
  const [sellerGroups, setSellerGroups] = useState({});


  const [showBackToTop, setShowBackToTop] = useState(false);

  const loadingTimerRef = useRef(null);
  const similarProductsRef = useRef(null);
  const searchInputRef = useRef(null);


const [displayedQuery, setDisplayedQuery] = useState('');



  useEffect(() => {
    if (searchResults.length > 0) {
      // Extract image URLs
      const imageUrls = searchResults
        .map(product => product.thumbnail_url)
        .filter(Boolean);
      
      // Preload images in the background
      import('./services/imageService').then(({ preloadImages }) => {
        preloadImages(imageUrls);
      });
    }
  }, [searchResults]);

  // Check for mobile viewport on mount and resize
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      
      // Only update state if the value has changed to avoid render loops
      if (mobile !== isMobile) {
        setIsMobile(mobile);
      }
    };
    
    // Set initial value
    handleResize();
    
    // Add resize listener
    window.addEventListener('resize', handleResize);
    
    // Clean up
    return () => window.removeEventListener('resize', handleResize);
  }, [isMobile]);
  
  // Handle side effects of mobile state changes
  useEffect(() => {
    // Close similar products popup on mobile
    if (isMobile && showSimilarProducts) {
      setShowSimilarProducts(false);
    }
    
    // Reset to results view when switching to mobile
    if (isMobile && tabView !== 'results') {
      setTabView('results');
    }
  }, [isMobile, showSimilarProducts, tabView]);

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
    if (isMobile) return; // Don't show previews on mobile
    setPreviewProduct(productId);
  };

  const handleProductPreviewLeave = () => {
    setPreviewProduct(null);
  };

  function RecentSearchesComponent({ searchHistory, setQuery, performSearch, darkMode }) {
    if (!searchHistory || searchHistory.length === 0) return null;
    
    return (
      <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} p-5 rounded-lg shadow-sm mb-6 border-2`}>
        <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-3 flex items-center`}>
          <Search className="w-4 h-4 mr-2 text-[#FE90EA]" />
          <span className={`${darkMode ? 'text-white' : 'text-black'} border-b-2 border-[#FE90EA] pb-1`}>Your Recent Searches</span>
        </h3>
        
        <div className="space-y-2 max-h-36 overflow-y-auto pr-1 custom-scrollbar">
          {searchHistory.map((item, index) => (
            <div 
              key={index}
              className={`${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-50 hover:bg-gray-100'} px-3 py-2 rounded-md cursor-pointer transition-colors flex justify-between items-center`}
              onClick={() => { setQuery(item.query); performSearch(item.query); }}
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

  function LoadingSpinner({ query, darkMode }) {
    return (
      <div className={`flex flex-col items-center justify-center ${darkMode ? 'bg-gray-800' : 'bg-white'} p-6 sm:p-12 rounded-lg shadow-sm text-center`}>
        <svg 
          className="animate-spin h-12 w-12 sm:h-16 sm:w-16 text-[#FE90EA] mb-4" 
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
  
  // Mobile navigation tabs
  function MobileNavigationTabs({ currentTab, setTabView, darkMode }) {
    return (
      <div className={`flex w-full mb-4 rounded-lg overflow-hidden ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>
        <button 
          className={`flex-1 py-2 text-center text-sm font-medium ${
            currentTab === 'results' 
              ? `${darkMode ? 'bg-gray-600 text-white' : 'bg-white text-gray-800'}`
              : `${darkMode ? 'text-gray-300' : 'text-gray-600'}`
          }`}
          onClick={() => setTabView('results')}
        >
          Results
        </button>
        <button 
          className={`flex-1 py-2 text-center text-sm font-medium ${
            currentTab === 'history' 
              ? `${darkMode ? 'bg-gray-600 text-white' : 'bg-white text-gray-800'}`
              : `${darkMode ? 'text-gray-300' : 'text-gray-600'}`
          }`}
          onClick={() => setTabView('history')}
        >
          History
        </button>
      </div>
    );
  }

  const firstQuery = queryExamples[Math.floor(Math.random() * queryExamples.length)];

  useEffect(() => {
    setQuery(firstQuery);
  }, []);

const initialSearchRef = useRef(false);

useEffect(() => {
  if (initialSearchRef.current) return; // Only run this once
  
  const randomQuery = queryExamples[Math.floor(Math.random() * queryExamples.length)];
  setQuery(randomQuery);
  
  // Use a timeout to ensure the query state is updated before search
  const timer = setTimeout(() => {
    console.log("Executing initial search for:", randomQuery);
    performSearch(randomQuery); // Call search directly with the value
    initialSearchRef.current = true; // Mark as executed
  }, 300);
  
  return () => clearTimeout(timer);
}, []); // Empty dependency array means this runs once on mount

const lastQueryRef = useRef({ text: '', timestamp: 0 });



  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    performSearch(query);
  };
  
  const performSearch = async (searchQuery) => {
    if (!searchQuery.trim()) return;
    
    // Log for debugging
    console.log("Performing search for:", searchQuery);
    
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
      }, 1000);
    }
    
    // Always switch to results tab when searching
    if (isMobile) {
      setTabView('results');
    }
    
    try {
      let data;
      // Use the passed searchQuery parameter, not the component state
      if(searchQuery[0] === "\"" && searchQuery[searchQuery.length - 1] === "\""){
        searchQuery = searchQuery.substring(1, searchQuery.length - 1);
        data = await searchProducts("exact_match", searchQuery);
      } else {
        data = await searchProducts(searchProfile, searchQuery);
      }
  
      // Log results for debugging
      console.log("Search results:", data.results ? data.results.length : 0);
    
      // Process the product images to add proxy URLs before setting state
      const processedResults = data.results && data.results.length > 0 
        ? processProductImages(data.results) 
        : [];
      
      // Make sure we're setting an array even if data.results is undefined
      setSearchResults(processedResults);
  
      // Update search history
      if (!isFirstSearch && data.results){
        setSearchHistory(prev => [
          { query: searchQuery, timestamp: new Date().toLocaleTimeString(), results: data.results.length || 0, queryTime: data.query_time_ms },
          ...prev.slice(0, 9)
        ]);
      }
    } catch (error) {
      console.error('Search error:', error);
      // In case of error, set empty results instead of keeping old results
      setSearchResults([]);
      
      // Fallback to mock data after a delay to simulate network request
      setTimeout(() => {
        const mockResults = Array(6).fill(0).map((_, i) => {
          // Generate random colors
          const bgColors = ['212121', '4a4a4a', '6b6b6b', '444', '333', '555', 'abd123', 'fe90ea', '256789', '742d1e'];
          const textColors = ['ffffff', 'f0f0f0', 'eeeeee', 'dddddd', 'cccccc'];
          
          // Select random colors from our arrays
          const bgColor = bgColors[Math.floor(Math.random() * bgColors.length)];
          const textColor = textColors[Math.floor(Math.random() * textColors.length)];
          
          return {
            id: `result-${i}`,
            score: (Math.random() * 0.2 + 0.8).toFixed(2),
            name: `${searchQuery.charAt(0).toUpperCase() + searchQuery.slice(1)} Product ${i + 1}`,
            description: `This is a sample product related to "${searchQuery}".`,
            thumbnail_url: `https://placehold.co/600x400/${bgColor}/${textColor}?text=${encodeURIComponent(searchQuery)}+${i+1}`,
            price_cents: Math.floor(Math.random() * 5000) + 1000,
            ratings_score: (Math.random() * 1 + 4).toFixed(1),
            ratings_count: Math.floor(Math.random() * 300) + 50,
            seller_name: `Seller ${i % 3 + 1}`,
            seller_id: `seller-${i % 3 + 1}`,
            url: '#'
          };
        });
        setSearchResults(mockResults);
      }, 1000);
    } finally {
      // IMPORTANT: Clear the spinner timer immediately
      if (loadingTimerRef.current) {
        clearTimeout(loadingTimerRef.current);
        loadingTimerRef.current = null;
      }
      
      setDisplayedQuery(searchQuery);
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

  const handleProductHover = useCallback(async (productOrSeller, event, isSeller = false) => {
    // Skip hover behavior on mobile
    if (isMobile) return;
    
    // Clear any existing hover timers
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
    }
    if (blindSpotTimerRef.current) {
      clearTimeout(blindSpotTimerRef.current);
    }
    
    // Set product as being hovered
    setIsProductHovered(true);
    setHoveredProduct(productOrSeller);
    
    // Get the dimensions and position of the card
    const card = event.currentTarget;
    const rect = card.getBoundingClientRect();
    
    // Store hover position
    // Make sure position is valid and doesn't go off screen
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
    const xPos = Math.min(rect.right + 10, viewportWidth - 310); // 300px + 10px margin
    
    setHoverPosition({ 
      x: xPos,
      y: rect.top 
    });
    
    // Different behavior based on whether it's a seller card or product card
    if (isSeller) {
      // For seller cards, we already have the products, so we use those directly
      const sellerProducts = productOrSeller.products || [];
      
      // Use all the products instead of limiting them
      // Process the images for consistent display
      const formattedProducts = sellerProducts
      .map(product => ({
        ...product,
        score: product.score || "1.00",
        thumbnail_url: product.thumbnail_url || generatePlaceholder(100,100,product.name)
      }))
      .sort((a, b) => {
        // First, we need to calculate mean and standard deviation for both metrics
        const calculateStats = (products, key) => {
          const values = products.map(p => p[key] || 0);
          const sum = values.reduce((acc, val) => acc + val, 0);
          const mean = sum / values.length;
          const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
          const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / values.length;
          const stdDev = Math.sqrt(variance);
          return { mean, stdDev };
        };

        const { mean: countMean, stdDev: countStdDev } = calculateStats(sellerProducts, 'ratings_count');
        const { mean: scoreMean, stdDev: scoreStdDev } = calculateStats(sellerProducts, 'ratings_score');

        // Calculate z-scores for both products
        const getZScore = (value, mean, stdDev) => {
          // Handle case of zero standard deviation
          if (stdDev === 0) return 0;
          return (value - mean) / stdDev;
        };

        const countZScoreA = getZScore(a.ratings_count || 0, countMean, countStdDev);
        const scoreZScoreA = getZScore(a.ratings_score || 0, scoreMean, scoreStdDev);
        
        const countZScoreB = getZScore(b.ratings_count || 0, countMean, countStdDev);
        const scoreZScoreB = getZScore(b.ratings_score || 0, scoreMean, scoreStdDev);

        // Combine z-scores (you can adjust weights if needed)
        const combinedZScoreA = countZScoreA + scoreZScoreA;
        const combinedZScoreB = countZScoreB + scoreZScoreB;

        // Sort descending by combined z-score
        return combinedZScoreB - combinedZScoreA;
      });
      
      setSimilarProducts(formattedProducts);
      
      // Set the title for the popup
      setSelectedProduct({
        ...productOrSeller,
        name: productOrSeller.name || "This Seller", // Use seller name
        isSeller: true // Flag to indicate this is a seller
      });
    } else {
      // This is a regular product, fetch similar products as before
      try {
        const data = await getSimilarProducts(productOrSeller.description, productOrSeller.name, productOrSeller.id);
        // Filter out the current product from results and process images
        const similarProducts = processProductImages(
          data.results.filter(item => item.name !== productOrSeller.name)
        ).map(item => ({
          ...item,
          score: parseFloat(item.score).toFixed(2) // Format score
        }));
            
        setSimilarProducts(similarProducts);
        setSelectedProduct(productOrSeller); // Store the hovered product
      } catch (error) {
        console.error('Error fetching similar products:', error);
        // Fallback to mock data
        const fakeSimilarProducts = Array(5).fill(0).map((_, i) => ({
          id: `similar-${i}`,
          name: `Similar (images) to ${productOrSeller.name} - Item ${i + 1}`,
          description: `A product similar to ${productOrSeller.name}.`,
          thumbnail_url: `https://placehold.co/50x50?text=Similar+${i+1}`,
          score: (Math.random() * 0.3 + 0.7).toFixed(2),
          ratings_count: Math.floor(Math.random() * 100) + 5,
          ratings_score: (Math.random() * 1 + 4).toFixed(1),
          price_cents: Math.floor(Math.random() * 5000) + 500,
          url: '#'
        }));
        
        setSimilarProducts(fakeSimilarProducts);
        setSelectedProduct(productOrSeller);
      }
    }
    
    // Set timer to show the similar products after a short delay
    hoverTimerRef.current = setTimeout(() => {
      setShowSimilarProducts(true);
      
      // Reset scroll position when showing popup
      if (similarProductsScrollRef.current) {
        similarProductsScrollRef.current.scrollTop = 0;
      }
    }, 150);
  }, [isMobile]);


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
    // Skip on mobile
    if (isMobile) return;
    
    setIsProductHovered(false);
    
    // Use a delay before closing to handle the blind spot
    blindSpotTimerRef.current = setTimeout(() => {
      // Only close if popup is not being hovered
      if (!isPopupHovered) {
        closeSimilarProducts();
      }
    }, 100); 
  }, [isPopupHovered, closeSimilarProducts, isMobile]);

  const handlePopupMouseLeave = useCallback(() => {
    // Skip on mobile
    if (isMobile) return;
    
    setIsPopupHovered(false);
    
    // Close after a short delay if product is not hovered
    blindSpotTimerRef.current = setTimeout(() => {
      if (!isProductHovered) {
        closeSimilarProducts();
      }
    }, 100);
  }, [isProductHovered, closeSimilarProducts, isMobile]);

  // Handle similar products popup mouse enter
  const handlePopupMouseEnter = () => {
    // Skip on mobile
    if (isMobile) return;
    
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


  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 300) {
        setShowBackToTop(true);
      } else {
        setShowBackToTop(false);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };


  // Group products by seller whenever search results change
  useEffect(() => {
    if (searchResults && searchResults.length > 0) {
      // Create seller groups from search results
      const groups = {};
      
      searchResults.forEach(product => {
        // Extract seller info from product
        const sellerId = product.seller?.id || 
                        (product.seller_name ? `seller-${product.seller_name}` : 'unknown');
        const sellerName = product.seller?.name || product.seller_name || 'Unknown Seller';
        
        if (!groups[sellerId]) {
          groups[sellerId] = {
            id: sellerId,
            name: sellerName,
            products: []
          };
        }
        
        groups[sellerId].products.push(product);
      });
      
      setSellerGroups(groups);
    } else {
      setSellerGroups({});
    }
  }, [searchResults]);


  return (
    <div className={`flex flex-col min-h-screen ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-black'}`}>
      {/* Header */}
      <header className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-sm py-3 sm:py-4 px-4 sm:px-6 border-b-2 border-[#FE90EA] main-header`}>
        <div className="mx-auto flex flex-wrap justify-between items-center">
          <div className="flex items-center space-x-2 mb-2 sm:mb-0">
            <img src="/gum.png" alt="Gum" className="h-6 sm:h-8 w-auto" />
            <h1 className={`text-base sm:text-xl font-bold ${darkMode ? 'text-white' : 'text-black'}`}>Search Prototype</h1>
          </div>
          
          {/* Middle section with nav links */}
          <div className="flex items-center space-x-3 sm:space-x-6 order-3 sm:order-2 w-full sm:w-auto justify-center mt-3 sm:mt-0">
            
            <a 
              href="https://www.notion.so/Search-Discovery-Case-Study-Blog-40e476a45ad94596ad323289eac62c2c" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center px-2 sm:px-3 py-1 text-xs font-medium text-black bg-[#FE90EA] rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-1 focus:ring-[#FE90EA] border border-black"
            >
              Case Study
            </a>
            <a 
              href="https://phileas.me" 
              target="_blank" 
              rel="noopener noreferrer"
              className={`text-xs sm:text-sm ${darkMode ? 'text-gray-300 hover:text-white' : 'text-gray-700 hover:text-black'} transition-colors flex items-center`}
            >
              <span className="mr-1">By</span>
              <span className="font-medium text-[#FE90EA]">Phileas Hocquard</span>
            </a>
          </div>
          
          <div className="flex items-center space-x-2 sm:space-x-4 ml-auto sm:ml-0 order-2 sm:order-3">
            <div className={`text-xs sm:text-sm ${darkMode ? 'text-gray-300' : 'text-black'} hidden sm:block`}>
              Products Indexed: 5,467
            </div>
            <div className={`text-xs ${darkMode ? 'text-gray-300' : 'text-black'} hidden md:block`}>
              Shards: 1
            </div>
            <SearchProfileSelector
              searchProfile={searchProfile}
              setSearchProfile={setSearchProfile}
              searchProfiles={searchProfiles}
              darkMode={darkMode}
            />
            {/* Dark mode toggle */}
            <button 
              onClick={toggleDarkMode} 
              className={`p-1 sm:p-2 rounded-full border-2 ${darkMode ? 'bg-gray-700 text-yellow-400 border-[#FE90EA]' : 'bg-gray-200 text-gray-700 border-black'}`}
              aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {darkMode ? <Sun className="h-4 w-4 sm:h-5 sm:w-5" /> : <Moon className="h-4 w-4 sm:h-5 sm:w-5" />}
            </button>
          </div>
        </div>
      </header>
        
      {/* Main content */}
      <main className="flex-grow py-3 sm:py-6 px-3 sm:px-6 page-content">
        <div className="w-full max-w-7xl mx-auto">
          {/* Search form with ScrollingQueryExamples to the right */}
          <div className="flex flex-col md:flex-row md:items-start w-full">
            <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} p-3 rounded-lg shadow-sm mb-4 border-2 w-full md:flex-grow sm:p-6 sm:mb-6`}>
              <form onSubmit={handleSearch} className="w-full search-form">
                <div className="flex flex-col w-full md:flex-row md:items-center md:gap-4">
                  <div className="relative flex-grow w-full mb-2 md:mb-0">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-[#FE90EA]" />
                    <input
                      ref={searchInputRef}
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Search for products..."
                      className={`w-full pl-10 pr-4 py-2 sm:py-3 rounded-md border-2 ${
                        darkMode
                          ? 'border-gray-600 bg-gray-700 text-white focus:border-[#FE90EA]'
                          : 'border-gray-300 bg-white text-black focus:border-[#FE90EA]'
                      } focus:outline-none focus:ring-1 focus:ring-[#FE90EA]`}
                      onClick={(e) => e.target.select()}
                    />
                  </div>
                  <button
                    type="submit"
                    className="w-full md:w-auto bg-[#FE90EA] text-black px-4 sm:px-6 py-2 sm:py-3 rounded-md hover:bg-[#ff9eef] focus:outline-none focus:ring-2 focus:ring-[#FE90EA] focus:ring-offset-2 font-medium border-2 border-black flex-shrink-0"
                    disabled={isLoading}
                  >
                    {isLoading ? 'Searching...' : 'Search'}
                  </button>
                </div>
              </form>
            </div>
            <div className="hidden md:block md:flex-shrink-0">
            <ScrollingQueryExamples
              setQuery={setQuery}
              performSearch={performSearch}
              darkMode={darkMode}
              queryExamples={queryExamples}
            />
          </div>
        </div>
      </div>

        {/* Mobile navigation tabs - only shown on mobile */}
        {/* {isMobile && (
          <MobileNavigationTabs 
            currentTab={tabView} 
            setTabView={setTabView} 
            darkMode={darkMode} 
          />
        )} */}
        {/* Two-column layout for desktop, stacked for mobile */}
        <div className="flex flex-col lg:flex-row gap-4 sm:gap-4">
          {/* Left column (wider) - Search results */}
          <div className={`${isMobile && tabView !== 'results' ? 'hidden' : 'block'} lg:w-2/3 space-y-4 sm:space-y-6`}>
            {/* Search results or loading state */}
            {isLoading && showLoadingSpinner ? (
              <LoadingSpinner darkMode={darkMode} query={query}/>
            ) : (
              <SearchResultsWithSellerFilter
                searchResults={searchResults}
                darkMode={darkMode}
                isLoading={isLoading}
                query={query}
                displayedQuery={displayedQuery} 
                onHover={handleProductHover}
                onLeave={handleProductMouseLeave}
                renderProductCard={(product, index) => (
                  <ProductCard 
                    key={`${product.id || product.name}-${index}`}
                    product={product}
                    index={index}
                    darkMode={darkMode}
                    onHover={handleProductHover}
                    onLeave={handleProductMouseLeave}
                  />
                )}
              />
            )}
            {/* Search history section - shown on desktop and mobile history tab */}
            {(!isMobile || tabView === 'history') && searchHistory.length > 0 && (
              <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-4 sm:p-6 rounded-lg shadow-sm`}>
                <h2 className={`text-base sm:text-lg font-semibold mb-3 sm:mb-4 ${darkMode ? 'text-white' : 'text-black'}`}>Recent Searches</h2>
                <div className="overflow-x-auto table-responsive">
                  <table className={`min-w-full divide-y ${darkMode ? 'divide-gray-700' : 'divide-gray-200'} history-table`}>
                    <thead className={darkMode ? 'bg-gray-700' : 'bg-gray-50'}>
                      <tr>
                        <th className={`px-3 sm:px-6 py-2 sm:py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Query</th>
                        <th className={`px-3 sm:px-6 py-2 sm:py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Time</th>
                        <th className={`px-3 sm:px-6 py-2 sm:py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Results</th>
                        <th className={`px-3 sm:px-6 py-2 sm:py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Query Time</th>
                        <th className={`px-3 sm:px-6 py-2 sm:py-3 text-left text-xs font-medium ${darkMode ? 'text-gray-300' : 'text-gray-500'} uppercase tracking-wider`}>Action</th>
                      </tr>
                    </thead>
                    <tbody className={`${darkMode ? 'bg-gray-800 divide-y divide-gray-700' : 'bg-white divide-y divide-gray-200'}`}>
                      {searchHistory.map((item, i) => (
                        <tr key={i} className={darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-50'}>
                          <td className={`px-3 sm:px-6 py-2 sm:py-4 whitespace-nowrap text-xs sm:text-sm font-medium ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>{item.query}</td>
                          <td className={`px-3 sm:px-6 py-2 sm:py-4 whitespace-nowrap text-xs sm:text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>{item.timestamp}</td>
                          <td className={`px-3 sm:px-6 py-2 sm:py-4 whitespace-nowrap text-xs sm:text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>{item.results}</td>
                          <td className={`px-3 sm:px-6 py-2 sm:py-4 whitespace-nowrap text-xs sm:text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>{item.queryTime?.toFixed(2) || '-'} ms</td>
                          <td className={`px-3 sm:px-6 py-2 sm:py-4 whitespace-nowrap text-xs sm:text-sm text-blue-500 hover:text-blue-700`}>
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
          
          {/* Right column - Hidden on mobile except in metrics tab */}
          <div className={`${isMobile && tabView !== 'metrics' ? 'hidden' : 'block'} lg:w-1/3`}>
            {/* Recent searches component - Only visible on desktop */}
            {!isMobile && (
              <RecentSearchesComponent 
                searchHistory={searchHistory.slice(0, 3)} 
                setQuery={setQuery} 
                performSearch={performSearch}
                darkMode={darkMode}
              />
            )}
          </div>
            
            </div>
      </main>

      {/* Footer */}
      <CopyrightFooter darkMode={darkMode} />
      
      {/* Similar products popup - not shown on mobile */}
      {!isMobile && showSimilarProducts && selectedProduct && (
        <div 
          ref={(el) => {
            similarProductsRef.current = el;
            similarProductsScrollRef.current = el;
          }}
          className={`fixed ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-xl p-2 z-50 border-2 border-[#FE90EA] custom-scrollbar`}
          style={{
            top: `${Math.max(hoverPosition.y, 10)-150}px`,
            left: `${hoverPosition.x}px`,
            width: '300px',
            maxHeight: '320px',
            overflowY: 'auto',
          }}
          onMouseEnter={handlePopupMouseEnter}
          onMouseLeave={handlePopupMouseLeave}
        >
          <div className="flex justify-between items-start pb-2">
            <h4 className={`font-small text-sm flex-grow pr-2 border-b-2 border-[#FE90EA] ${darkMode ? 'text-white' : 'text-black'}`}>
              {selectedProduct.isSeller 
                ? `Products from ${selectedProduct.name}`
                : `Similar items to "${selectedProduct.name.substring(0, 13)}${selectedProduct.name.length > 13 ? '...' : ''}"`}
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
                <a href={product.url || "#"} target="_blank" rel="noopener noreferrer" key={product.id || index} className="block">
                  <div 
                    className={`flex items-start py-2 ${darkMode ? 'border-gray-700 hover:bg-gray-700' : 'border-gray-100 hover:bg-gray-50'} border-b last:border-0 transition-colors rounded-md px-2`}
                  >
                    {/* Product Image */}
                    <div className="w-16 h-16 bg-gray-100 rounded-md overflow-hidden flex-shrink-0">
                      <img 
                        src={product.thumbnail_url || generatePlaceholder(100,100, product.name)} 
                        alt={product.name} 
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.target.src = generatePlaceholder(100,100, product.name);
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
                      
                      {/* Score - Show for similar products, but not for seller products */}
                      {!selectedProduct.isSeller && product.score && (
                        <div className="mt-2">
                          <div className="text-xs px-2 py-0.5 bg-[#FE90EA] text-black rounded-full inline-block">
                            Similarity: {parseFloat(product.score).toFixed(2)}
                          </div>
                        </div>
                      )}
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
                Loading products...
              </div>
            )}
          </div>
        </div>
      )}
      {/* Back to top button */}
      {showBackToTop && (
        <button
          onClick={scrollToTop}
          className={`fixed bottom-6 right-6 p-3 rounded-full shadow-lg z-50 text-black bg-[#FE90EA] hover:bg-[#ff9eef] border-2 border-black focus:outline-none`}
          aria-label="Back to top"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="19" x2="12" y2="5"></line>
            <polyline points="5 12 12 5 19 12"></polyline>
          </svg>
        </button>
      )}
    </div>
  );
}

// Updated ScrollingQueryExamples Component for better mobile display
function ScrollingQueryExamples({ setQuery, performSearch, darkMode, queryExamples}) {
  // Sample query examples
  
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
    performSearch(query);
  };
  
  return (
    <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} ml-3 sm:w-52 sm:p-3 rounded-lg shadow-sm mb-4 sm:mb-6 border-2 overflow-hidden`}>
      <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2 flex items-center`}>
        {/* <Search className="w-4 h-4 mr-2 text-[#FE90EA]" /> */}
        <span className={`${darkMode ? 'text-white' : 'text-black'} border-b-2 border-[#FE90EA] pb-1`}>Popular Queries</span>
      </h3>
      
      <div className="relative h-8 sm:h-10 overflow-hidden">
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
              className={`font-medium text-sm sm:text-base ${darkMode ? 'text-gray-200 hover:text-[#FE90EA]' : 'text-gray-800 hover:text-[#FE90EA]'} cursor-pointer truncate`}
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

function CopyrightFooter({ darkMode }) {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className={`mt-6 sm:mt-12 py-4 sm:py-6 border-t ${darkMode ? 'border-gray-700 text-gray-400' : 'border-gray-200 text-gray-500'}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <div className="flex items-center flex-wrap justify-center md:justify-start">
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
        
        <div className={`text-xs mt-4 text-center md:text-left ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          This application is for demonstration purposes only. Gumroad is a trademark of Gumroad, Inc.
          Search interface is not affiliated with, endorsed by, or sponsored by Gumroad.
        </div>
      </div>
    </footer>
  );
}

const generatePlaceholder = (dim1, dim2, title) => {
  const bgColors = ['212121', '4a4a4a', '6b6b6b', '444', '333', '555', 'abd123', 'fe90ea', '256789', '742d1e'];
  const textColors = ['ffffff', 'f0f0f0', 'eeeeee', 'dddddd', 'cccccc'];

  const bgColor = bgColors[Math.floor(Math.random() * bgColors.length)];
  const textColor = textColors[Math.floor(Math.random() * textColors.length)];


  return `https://placehold.co/${dim1}x${dim2}/${bgColor}/${textColor}?text=${title}`
}

export default App;