// SearchProfileSelector.jsx
import React, { useState } from 'react';
import { BarChartIcon } from 'lucide-react';

function SearchProfileSelector({ searchProfile, setSearchProfile, searchProfiles, darkMode }) {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  
  const handleProfileChange = (profileId) => {
    setSearchProfile(profileId);
    setIsDropdownOpen(false);
  };
  
  // Get current profile display name
  const currentProfile = searchProfiles.find(p => p.id === searchProfile) || { name: 'Select Profile' };
  
  return (
    <div className="relative">
      <button
        className={`flex items-center space-x-1 px-2 py-2 rounded-md ${
          darkMode 
            ? 'hover:bg-gray-700 text-gray-200' 
            : 'hover:bg-gray-100 text-gray-800'
        }`}
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
      >
        <BarChartIcon className="w-4 h-4 text-[#FE90EA]" />
        <span className="truncate max-w-[180px] sm:max-w-[220px]">
          {currentProfile.name} {currentProfile.version || ''}
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

      {isDropdownOpen && (
        <div 
          className={`absolute z-50 mt-1 w-64 rounded-md shadow-lg ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
          } border-2 py-1 custom-scrollbar max-h-64 overflow-y-auto right-0`}
        >
          <div className="p-2 border-b border-gray-200 dark:border-gray-700">
            <h3 className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} flex items-center`}>
              <BarChartIcon className="w-4 h-4 mr-2 text-[#FE90EA]" />
              <span className={`${darkMode ? 'text-white' : 'text-black'}`}>Search Profiles</span>
            </h3>
          </div>
          
          {searchProfiles.map((profile) => (
            <div
              key={profile.id}
              className={`px-4 py-2 text-sm cursor-pointer ${
                profile.id === searchProfile
                  ? `${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`
                  : `${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`
              } ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}
              onClick={() => handleProfileChange(profile.id)}
            >
              <div className="flex items-center">
                {profile.id === searchProfile && (
                  <svg className="w-3 h-3 text-[#FE90EA] mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
                <span className={profile.id === searchProfile ? 'font-medium' : ''}>
                  {profile.name} {profile.version && profile.version}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default SearchProfileSelector;