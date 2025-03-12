// src/services/imageQueue.js

// Simple image loading queue with proper exports
const loadQueue = [];
let isLoading = false;
let observerInitialized = false;
const observedElements = new Set();

// Add image to queue with priority
export function queueImageLoad(imgElement, priority = 'normal') {
  if (!imgElement) return;
  
  // Higher priority items go to front of queue
  if (priority === 'high') {
    loadQueue.unshift(imgElement);
  } else {
    loadQueue.push(imgElement);
  }
  
  // Start processing queue if not already processing
  if (!isLoading) {
    processNextImage();
  }
}

// Process next image in queue
function processNextImage() {
  if (loadQueue.length === 0) {
    isLoading = false;
    return;
  }
  
  isLoading = true;
  const img = loadQueue.shift();
  
  // Check if element is still in DOM and has data-src
  if (!img || !img.dataset || !img.dataset.src || !document.body.contains(img)) {
    // Skip this image and process next
    processNextImage();
    return;
  }
  
  const src = img.dataset.src;
  img.onload = img.onerror = () => {
    // Mark as loaded
    img.dataset.loaded = 'true';
    img.classList.add('image-loaded');
    
    // Small delay between images to prevent network saturation
    setTimeout(processNextImage, 100);
  };
  
  // Actually set the src to trigger loading
  img.src = src;
  // Remove the data-src to avoid loading again
  delete img.dataset.src;
}

// Initialize intersection observer for visibility detection
export function initializeObserver() {
  // Only initialize once
  if (observerInitialized) return;
  
  // Check browser support
  if (!('IntersectionObserver' in window)) {
    console.log('IntersectionObserver not supported, loading all images directly');
    return;
  }
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const card = entry.target;
      const img = card.querySelector('img[data-src]');
      
      if (!img) return;
      
      if (entry.isIntersecting) {
        // Calculate priority based on position
        const rect = card.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        const isFullyVisible = rect.top >= 0 && rect.bottom <= viewportHeight;
        
        // Prioritize fully visible products
        queueImageLoad(img, isFullyVisible ? 'high' : 'normal');
        observer.unobserve(card);
        observedElements.delete(card);
      }
    });
  }, { 
    rootMargin: '200px', // Load images slightly before they become visible
    threshold: 0.1 // Trigger when at least 10% of the element is visible
  });
  
  observerInitialized = true;
  
  // Expose observer for later use
  window.productImageObserver = observer;
  
  return observer;
}

// Observe a single product card element
export function observeProductCard(cardElement) {
  if (!cardElement || observedElements.has(cardElement)) return;
  
  if (!observerInitialized) {
    initializeObserver();
  }
  
  if (window.productImageObserver) {
    window.productImageObserver.observe(cardElement);
    observedElements.add(cardElement);
  }
}

// Function to reset and re-observe all product cards (call this when results update)
export function refreshProductCardObserver() {
  // Find all product cards and observe them
  const productCards = document.querySelectorAll('.product-card');
  
  if (!observerInitialized) {
    initializeObserver();
  }
  
  if (window.productImageObserver) {
    // Clear previously observed elements
    observedElements.forEach(element => {
      window.productImageObserver.unobserve(element);
    });
    observedElements.clear();
    
    // Observe new elements
    productCards.forEach(card => {
      window.productImageObserver.observe(card);
      observedElements.add(card);
    });
    
    console.log(`Observing ${productCards.length} product cards for visibility-based loading`);
  }
}

// Make sure to export all functions
export default {
  queueImageLoad,
  initializeObserver,
  observeProductCard,
  refreshProductCardObserver
};