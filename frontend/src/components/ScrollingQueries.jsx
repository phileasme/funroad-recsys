// NavbarScrollingQueries.jsx - Adapted from your existing ScrollingQueryExamples
import React, { useState, useRef, useEffect } from 'react';
import { Search } from 'lucide-react';

function ScrollingQueries({ setQuery, performSearch, darkMode }) {
  // Sample query examples - using the same data from your original component
  const queryExamples = ["ios 14 app icons", "kdp cover design", "python for beginners", "macbook air mockup", "ios 14 icons", "procreate brush pack", "mtt sng cash", "cross stitch pattern", "windows 11 themes", "max for live", "forex expert advisor", "figma ui kit", "kdp book cover", "cross stitch pdf", "ready to render", "macbook pro mockup", "ableton live packs", "kdp digital design", "royalty free music", "mt4 expert advisor", "sample pack", "betting system", "phone wallpaper", "design system", "tennis lessons", "poker online"];
  
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
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
    setIsDropdownOpen(false);
  };
  
  return (
    <div className="relative">
      {/* Button that shows current example and toggles dropdown */}
      <button
        className={`flex items-center space-x-1 px-3 py-2 rounded-md ${
          darkMode 
            ? 'hover:bg-gray-700 text-gray-200' 
            : 'hover:bg-gray-100 text-gray-800'
        }`}
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
      >
        <Search className="w-4 h-4 text-[#FE90EA]" />
        <span className="truncate max-w-[120px] sm:max-w-[150px]">
          "{queryExamples[currentIndex]}"
        </span>
        <svg 
          width="12" 
          height="12" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round"
          className={`transition-transform duration-200 ${isDropdownOpen ? 'rotate-180' : ''}`}
        >
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </button>

      {/* Dropdown with all query examples */}
      {isDropdownOpen && (
        <div 
          className={`absolute z-50 mt-1 w-64 rounded-md shadow-lg ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
          } border-2 py-1 custom-scrollbar max-h-64 overflow-y-auto`}
        >
          <div className="p-2 border-b border-gray-200 dark:border-gray-700">
            <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} flex items-center`}>
              <Search className="w-4 h-4 mr-2 text-[#FE90EA]" />
              <span className={`${darkMode ? 'text-white' : 'text-black'}`}>Popular Queries</span>
            </h3>
          </div>
          
          {queryExamples.map((query, index) => (
            <div
              key={index}
              className={`px-4 py-2 text-sm cursor-pointer ${
                darkMode 
                  ? 'hover:bg-gray-700 text-gray-200' 
                  : 'hover:bg-gray-100 text-gray-800'
              }`}
              onClick={() => handleQueryClick(query)}
            >
              <div className="flex items-center">
                <Search className="w-3 h-3 text-[#FE90EA] mr-2" />
                <span>"{query}"</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default NavbarScrollingQueries;