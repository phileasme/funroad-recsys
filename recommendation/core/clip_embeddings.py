from transformers import CLIPModel, CLIPProcessor
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Optional, Tuple
from pathlib import Path
import io
import base64
from functools import lru_cache
import logging
from dataclasses import dataclass
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingMetadata:
    embedding: np.ndarray
    processing_time: float
    input_type: str  # 'text' or 'image'
    original_size: Optional[Tuple[int, int]] = None  # For images only
    token_count: Optional[int] = None  # For text only

class CLIPEmbedding: #@TODO upgrade to siglip..
    
    def __init__(self, model_path:str=None, model_name: str = None):
        # Determine the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set default model name if not provided
        self.model_name = model_name or "openai/clip-vit-base-patch32"
        
        # Log our device and model information
        logger.info(f"Initializing CLIP with model: {self.model_name} on device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Always download from HuggingFace
        try:
            logger.info(f"Loading model from HuggingFace: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            logger.info("Successfully loaded CLIP model and processor")
            
            # Optionally save the model locally for faster reloading
            if model_path:
                try:
                    logger.info(f"Saving model to local path: {model_path}")
                    os.makedirs(model_path, exist_ok=True)
                    self.model.save_pretrained(model_path)
                    self.processor.save_pretrained(model_path)
                    logger.info(f"Model saved to {model_path}")
                except Exception as save_error:
                    logger.error(f"Could not save model locally: {save_error}")
        
        except Exception as download_error:
            logger.error(f"Failed to download model: {download_error}")
            raise
            
        # Set constants
        self.max_text_length = 77  # CLIP's default max token length
        self.image_size = 224  # CLIP's default image size
        
    @lru_cache(maxsize=1024)
    def get_text_embedding(self, text: str) -> EmbeddingMetadata:
        """
        Generate embedding for text input with caching
        
        Args:
            text: Input text string
            
        Returns:
            EmbeddingMetadata containing embedding and metadata
        """
        start_time = time.time()
        try:
            # Preprocess text
            text = ' '.join(text.split())  # Normalize whitespace
            
            chunks = [text[i:i+300] for i in range(0, len(text), 300)]
            
            with torch.no_grad():
                inputs = self.processor(
                    text=chunks[0],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model.get_text_features(**inputs)
                embedding = outputs.cpu().numpy()
                
                # Get token count
                token_count = len(self.processor.tokenizer.encode(text))
                
                processing_time = time.time() - start_time
                
                return EmbeddingMetadata(
                    embedding=embedding,
                    processing_time=processing_time,
                    input_type='text',
                    token_count=token_count
                )
                
        except Exception as e:
            logger.error(f"Error in text embedding: {e}")
            raise

    def get_image_embedding(
        self,
        image: Union[str, Path, Image.Image, bytes],
        base64_string: bool = False
    ) -> EmbeddingMetadata:
        """
        Generate embedding for image input
        
        Args:
            image: Input image as file path, PIL Image, bytes, or base64 string
            base64_string: Whether the input is a base64 string
            
        Returns:
            EmbeddingMetadata containing embedding and metadata
        """
        start_time = time.time()
        try:
            # Convert input to PIL Image
            if base64_string:
                if isinstance(image, bytes):
                    image = image.decode('utf-8')
                image_data = base64.b64decode(image)
                pil_image = Image.open(io.BytesIO(image_data))
            elif isinstance(image, (str, Path)):
                pil_image = Image.open(image)
            elif isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image))
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ValueError("Unsupported image input type")
            
            # Store original size
            original_size = pil_image.size
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            with torch.no_grad():
                inputs = self.processor(
                    images=pil_image,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy()
                
                processing_time = time.time() - start_time
                
                return EmbeddingMetadata(
                    embedding=embedding,
                    processing_time=processing_time,
                    input_type='image',
                    original_size=original_size
                )
                
        except Exception as e:
            logger.error(f"Error in image embedding: {e}")
            raise

    def get_batch_embeddings(
        self,
        inputs: List[Union[str, Path, Image.Image, bytes]],
        input_type: str,
        batch_size: int = 32,
        base64_string: bool = False
    ) -> List[EmbeddingMetadata]:
        """
        Generate embeddings for a batch of inputs
        
        Args:
            inputs: List of inputs (text strings or images)
            input_type: Type of input ('text' or 'image')
            batch_size: Size of batches for processing
            base64_string: Whether image inputs are base64 strings
            
        Returns:
            List of EmbeddingMetadata for each input
        """
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            if input_type == 'text':
                batch_results = [self.get_text_embedding(text) for text in batch]
            else:  # image
                batch_results = [
                    self.get_image_embedding(img, base64_string) 
                    for img in batch
                ]
            
            results.extend(batch_results)
            
        return results

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        """
        return float(
            np.dot(embedding1, embedding2) / 
            (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        )

# Example usage
def main():
    # Initialize embedding model
    clip_embed = CLIPEmbedding()
    
    # Text embedding example
    text = "an elephant"
    text_embedding = clip_embed.get_text_embedding(text)
    print(f"\nText Embedding Stats:")
    print(f"Processing Time: {text_embedding.processing_time:.3f}s")
    print(f"Token Count: {text_embedding.token_count}")
    print(f"Embedding Shape: {text_embedding.embedding.shape}")
    
    # Image embedding example
    image_path = "./core/test_images/guy_elph.jpg"  # Path to your image
    if Path(image_path).exists():
        image_embedding = clip_embed.get_image_embedding(image_path)
        print(f"\nImage Embedding Stats:")
        print(f"Processing Time: {image_embedding.processing_time:.3f}s")
        print(f"Original Size: {image_embedding.original_size}")
        print(f"Embedding Shape: {image_embedding.embedding.shape}")
        
        # Compute similarity
        similarity = clip_embed.compute_similarity(
            text_embedding.embedding[0],
            image_embedding.embedding[0]
        )
        print(f"\nText-Image Similarity: {similarity:.3f}")

if __name__ == "__main__":
    main()