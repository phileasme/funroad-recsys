from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import Union, List, Optional, Tuple, Dict
import logging
from dataclasses import dataclass
import time
import os
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed for reproducibility
set_seed()

@dataclass
class ColBERTEmbeddingMetadata:
    embedding: np.ndarray
    attention_mask: np.ndarray
    processing_time: float
    token_count: Optional[int] = None
    
@dataclass
class EmbeddingMetadata:
    embedding: np.ndarray
    processing_time: float
    input_type: str  # 'text' or 'image'
    original_size: Optional[Tuple[int, int]] = None  # For images only
    token_count: Optional[int] = None  # For text only

class ColBERTEmbedding:

    def __init__(self, model_path: str = None, model_name: str = "colbert-ir/colbertv2.0"):
        """Initialize model for embeddings"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Log our device and model information
        logger.info(f"Initializing ColBERT with model: {model_name} on device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Always download from HuggingFace
        try:
            logger.info(f"Loading model from HuggingFace: {model_name}")
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Successfully loaded ColBERT model and tokenizer")
            
            # Optionally save the model locally for faster reloading
            if model_path:
                try:
                    logger.info(f"Saving model to local path: {model_path}")
                    os.makedirs(model_path, exist_ok=True)
                    self.model.save_pretrained(model_path)
                    self.tokenizer.save_pretrained(model_path)
                    logger.info(f"Model saved to {model_path}")
                except Exception as save_error:
                    logger.error(f"Could not save model locally: {save_error}")
        
        except Exception as download_error:
            logger.error(f"Failed to download model: {download_error}")
            raise

        # Set model to evaluation mode
        self.model.eval()
        
        # Set constants
        self.max_seq_length = 512
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Using full embedding dimension: {self.embedding_dim}")
        
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text for ColBERT"""
        # Truncate text if it's too long
        truncated_text = text[:300]  # Limit text length before tokenization
        
        return self.tokenizer(
            truncated_text,
            padding="max_length",
            truncation=True,
            max_length=77,  # Explicitly set to 77
            return_tensors="pt"
        )
 
    @lru_cache(maxsize=1024)
    def get_colbert_sentence_embedding(self, text: str) -> EmbeddingMetadata:
        """
        Generate a normalized sentence-level embedding from token-level embeddings
        
        Args:
            text: Input text string
            
        Returns:
            Normalized sentence embedding vector
        """
        start_time = time.time()
        # Get the token-level embeddings
        token_embedding_metadata = self.get_colbert_word_embedding(text)
        
        # Get the embeddings and attention mask
        token_embeddings = token_embedding_metadata.embedding
        attention_mask = token_embedding_metadata.attention_mask
        
        # Create a mask for valid tokens (where attention_mask is 1)
        valid_token_mask = attention_mask[0] == 1
        
        # Extract only the embeddings of valid tokens
        valid_embeddings = token_embeddings[0][valid_token_mask]
        
        # Compute the mean of valid token embeddings
        sentence_embedding = np.mean(valid_embeddings, axis=0)
        
        # Normalize the sentence embedding (L2 normalization)
        normalized_sentence_embedding = sentence_embedding / np.linalg.norm(sentence_embedding)

        normalized_sentence_embedding = normalized_sentence_embedding.reshape(1,-1)

        processing_time = time.time() - start_time
        
        return EmbeddingMetadata(normalized_sentence_embedding, processing_time, "text", token_count=np.sum(valid_token_mask))

    @lru_cache(maxsize=1024)
    def get_colbert_word_embedding(self, text: str) -> ColBERTEmbeddingMetadata:
        """
        Generate ColBERT embedding for text input with caching
        # Pretty much the entire point of colbert is to use it as sparse vector field, where we maintain the encoding of each word.
        Args:
            text: Input text string
            
        Returns:
            ColBERTEmbeddingMetadata containing embedding and metadata
        """
        start_time = time.time()
        try:
            # Preprocess text
            text = ' '.join(text.split())  # Normalize whitespace
            
            with torch.no_grad():
                # Tokenize input
                inputs = self._tokenize(text)
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get token count
                token_count = torch.sum(inputs['attention_mask']).item()
                
                # Get token embeddings from BERT
                outputs = self.model(**inputs)
                
                # Get last hidden state
                last_hidden = outputs.last_hidden_state
                
                # Apply attention mask to zero out padding tokens
                masked_embeddings = last_hidden * inputs['attention_mask'].unsqueeze(-1)
                
                # Skip projection and keep the full embedding dimensions
                
                # Normalize each token embedding (L2 normalization)
                normalized_embeddings = torch.nn.functional.normalize(masked_embeddings, p=2, dim=2)
                
                # Convert to numpy
                embeddings_np = normalized_embeddings.cpu().numpy()
                attention_mask_np = inputs['attention_mask'].cpu().numpy()
                
                processing_time = time.time() - start_time
                
                return ColBERTEmbeddingMetadata(
                    embedding=embeddings_np,
                    attention_mask=attention_mask_np,
                    processing_time=processing_time,
                    token_count=token_count
                )
                
        except Exception as e:
            logger.error(f"Error in ColBERT embedding: {e}")
            raise
        
    def get_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> List[EmbeddingMetadata]:
        """
        Generate embeddings for a batch of text inputs
        
        Args:
            texts: List of text strings
            batch_size: Size of batches for processing
            
        Returns:
            List of EmbeddingMetadata for each input
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            batch_results = [self.get_colbert_sentence_embedding(text) for text in batch]
            
            results.extend(batch_results)
            
        return results
    
    def compute_token_similarity(
        self,
        query_embedding: np.ndarray,
        document_embedding: np.ndarray,
        query_mask: Optional[np.ndarray] = None,
        doc_mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute similarity between query and document embeddings using ColBERT's late interaction
        
        Args:
            query_embedding: ColBERT embedding for query
            document_embedding: ColBERT embedding for document
            query_mask: Attention mask for query tokens (optional)
            doc_mask: Attention mask for document tokens (optional)
            
        Returns:
            Similarity score
        """
        # Convert to torch tensors if they're numpy arrays
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).to(self.device)
        
        if isinstance(document_embedding, np.ndarray):
            document_embedding = torch.from_numpy(document_embedding).to(self.device)
        
        # Get shapes for easier reference    
        batch_q, len_q, dim_q = query_embedding.shape
        batch_d, len_d, dim_d = document_embedding.shape
        
        # Create default masks if not provided
        if query_mask is None:
            query_mask = torch.ones(batch_q, len_q, device=self.device)
        elif isinstance(query_mask, np.ndarray):
            query_mask = torch.from_numpy(query_mask).to(self.device)
        
        if doc_mask is None:
            doc_mask = torch.ones(batch_d, len_d, device=self.device)
        elif isinstance(doc_mask, np.ndarray):
            doc_mask = torch.from_numpy(doc_mask).to(self.device)
            
        # Calculate similarity matrix: batch x query_len x doc_len
        similarity_matrix = torch.matmul(
            query_embedding, 
            document_embedding.transpose(1, 2)
        )
        
        # Apply MaxSim: For each query token, find the maximum similarity with any document token
        max_similarities, _ = similarity_matrix.max(dim=2)  # batch x query_len
        
        # Apply query attention mask
        masked_max_similarities = max_similarities * query_mask
        
        # Sum the maximum similarities (only for non-padding tokens)
        score = masked_max_similarities.sum().item()
        
        # Option: Normalize by the number of query tokens for more consistent scoring
        num_query_tokens = query_mask.sum().item()
        normalized_score = score / (num_query_tokens if num_query_tokens > 0 else 1)
        
        return normalized_score

# Example usage
def main():
    # Initialize embedding model
    colbert_embed = ColBERTEmbedding()
    
    # Text embedding example
    query = "poker"
    document = "World tour poker player"
    
    # Get embeddings
    query_embedding = colbert_embed.get_colbert_word_embedding(query)
    doc_embedding = colbert_embed.get_colbert_word_embedding(document)
    
    print(f"\nQuery Embedding Stats:")
    print(f"Processing Time: {query_embedding.processing_time:.3f}s")
    print(f"Token Count: {query_embedding.token_count}")
    print(f"Embedding Shape: {query_embedding.embedding.shape}")
    
    print(f"\nDocument Embedding Stats:")
    print(f"Processing Time: {doc_embedding.processing_time:.3f}s")
    print(f"Token Count: {doc_embedding.token_count}")
    print(f"Embedding Shape: {doc_embedding.embedding.shape}")

    query_sentence_embedding = colbert_embed.get_colbert_sentence_embedding(query)
    document_sentence_embedding = colbert_embed.get_colbert_sentence_embedding(document)
    print("Sentence embedding shape:", query_sentence_embedding.embedding.shape)
    print("Document embedding shape:", document_sentence_embedding.embedding.shape)
    print("Sentence similarity (dot product):", document_sentence_embedding.embedding.dot(query_sentence_embedding.embedding.T))
    
    # Compute similarity with proper normalization
    similarity = colbert_embed.compute_token_similarity(
        query_embedding.embedding,
        doc_embedding.embedding,
        query_mask=query_embedding.attention_mask,
        doc_mask=doc_embedding.attention_mask
    )
    
    print(f"\nQuery-Document Similarity (Normalized): {similarity:.3f}")
    
    # For debugging, print number of actual tokens used in comparison
    print(f"Query tokens considered: {query_embedding.attention_mask.sum()}")
    print(f"Document tokens considered: {doc_embedding.attention_mask.sum()}")

if __name__ == "__main__":
    main()