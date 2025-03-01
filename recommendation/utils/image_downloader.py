import tempfile
import os
import requests
from PIL import Image
import io
from urllib.parse import urlparse
import logging
from typing import Optional, Union
import time

class ImageDownloader:
    # Few areas could be improved, like async gather download tmp images.
    def __init__(self, timeout: int = 10, max_retries: int = 3, retry_delay: int = 1):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_image(self, url: str) -> Optional[str]:
        """
        Downloads an image from a URL to a temporary file and returns the file path.
        
        Args:
            url (str): The URL of the image to download
            
        Returns:
            Optional[str]: Path to temporary file containing the image, or None if download failed
            
        Note:
            The temporary file will be automatically deleted when closed or when the program exits
        """
        if not url:
            self.logger.error("No URL provided")
            return None

        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                self.logger.error(f"Invalid URL format: {url}")
                return None
        except Exception as e:
            self.logger.error(f"Error parsing URL {url}: {e}")
            return None

        # Create a temporary file
        try:
            # suffix will be preserved, helping identify file type
            suffix = os.path.splitext(urlparse(url).path)[1]
            if not suffix:
                suffix = '.jpg'  # default extension if none found
                
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_path = temp_file.name
            
            # Try to download with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(url, timeout=self.timeout, stream=True)
                    response.raise_for_status()
                    
                    # Verify it's an image by trying to open it
                    img = Image.open(io.BytesIO(response.content))
                    
                    # Write to temporary file
                    temp_file.write(response.content)
                    temp_file.flush()
                    
                    self.logger.info(f"Successfully downloaded image to {temp_path}")
                    return temp_path
                    
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing image from {url}: {e}")
                    self._cleanup_temp_file(temp_path)
                    return None
                    
            self.logger.error(f"Failed to download image after {self.max_retries} attempts")
            self._cleanup_temp_file(temp_path)
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating temporary file: {e}")
            return None

    def _cleanup_temp_file(self, filepath: str) -> None:
        """
        Safely removes a temporary file if it exists
        
        Args:
            filepath (str): Path to the file to remove
        """
        try:
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
                self.logger.info(f"Cleaned up temporary file: {filepath}")
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary file {filepath}: {e}")

def process_image_url(url: str) -> Optional[Union[str, Image.Image]]:
    """
    Convenience function to download and process an image URL
    
    Args:
        url (str): The URL of the image to process
        
    Returns:
        Optional[Union[str, Image.Image]]: Path to temporary file containing the image,
                                         or None if processing failed
    """
    downloader = ImageDownloader()
    return downloader.download_image(url)