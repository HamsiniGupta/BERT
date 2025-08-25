from typing import List, Dict
import torch
import numpy as np

# =====================

from weaviate.classes.config import Configure, Property, DataType, VectorDistances
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

# =====================

from langchain_core.documents import Document
from transformers import AutoModel, AutoTokenizer

# =====================

class BERTEmbeddings:
    """BERT embeddings for Weaviate integration"""
    
    def __init__(self, model_name: str = "google-bert/bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"BERT model loaded on {self.device}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts to embeddings"""
        embeddings = []
        
        # Process in batches to manage memory
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts"""
        # Tokenize all texts in the batch
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return [emb for emb in embeddings]
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        return embedding


class WeaviateManager:
    
    def __init__(self, url: str, api_key: str, hf_token: str, bert_model_name: str = "google-bert/bert-base-uncased"):
        """Initialize Weaviate client with BERT embeddings."""
        self.url = url
        self.api_key = api_key
        self.hf_token = hf_token
        self.bert_model_name = bert_model_name
        
        # Initialize Weaviate client
        self._connect_to_weaviate()

        # Device config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize BERT embeddings model
        print("Loading BERT model for query encoding...")
        self.bert_embeddings = BERTEmbeddings(bert_model_name)
        print("BERT model loaded successfully!")

    def _connect_to_weaviate(self):
        """Connect to Weaviate using v4 client syntax"""
        
        print(f"Connecting to Weaviate at {self.url}")
        
        # Strategy 1: Try with API Key (most reliable for Weaviate Cloud)
        try:
            print("Trying API Key authentication...")
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=Auth.api_key(self.api_key),
                headers={"X-HuggingFace-Api-Key": self.hf_token} if self.hf_token else None,
            )
            
            # Test connection
            if self.client.is_ready():
                print("Connected successfully with API Key authentication")
                return
            else:
                print("Client not ready")
                self.client.close()
                
        except Exception as e:
            print(f"API Key auth failed: {e}")
            try:
                self.client.close()
            except:
                pass
        

    def reindex_with_embeddings(self, documents, embeddings_model):
        """Reindex documents using BERT embeddings"""
        print("Reindexing with BERT embeddings...")
        
        collection_name = "PMQA_Bert" 
        
        # Delete existing collection if it exists
        if self.client.collections.exists(collection_name):
            print(f"Deleting existing collection: {collection_name}")
            self.client.collections.delete(collection_name)
        
        # Create new collection for BERT
        print(f"Creating collection: {collection_name}")
        self.client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),  # We'll provide our own vectors
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="context_id", data_type=DataType.TEXT),
            ]
        )
        
        collection = self.client.collections.get(collection_name)
        
        print(f"Encoding {len(documents)} documents with BERT...")
        
        # Process documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Extract text content
            texts = [doc.page_content for doc in batch_docs]
            
            # Get embeddings from BERT model
            embeddings = self.bert_embeddings.encode(texts)
            
            # Use batch insertion with proper format
            with collection.batch.dynamic() as batch:
                for j, doc in enumerate(batch_docs):
                    # Extract metadata fields safely
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    
                    batch.add_object(
                        properties={
                            "content": doc.page_content,
                            "source": str(metadata.get("source", "")),
                            "document_id": str(metadata.get("document_id", "")),
                            "context_id": str(metadata.get("context_id", "")),
                        },
                        vector=embeddings[j].tolist()
                    )
            
            print(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
        
        print(f"Successfully indexed {len(documents)} documents with BERT!")
        return collection_name

    # Legacy method name for compatibility
    def reindex_with_simcse(self, documents, model_path=None):
        """Legacy method - now uses BERT instead of SimCSE"""
        print("Warning: reindex_with_simcse is deprecated. Using BERT embeddings instead.")
        return self.reindex_with_embeddings(documents, self.bert_embeddings)
            
    def create_schema(self):
        schema = "PMQA_Bert"
        if self.client.collections.exists(schema):
            print("Schema already exists")
            return

        self.client.collections.create(
            schema,
            vectorizer_config=Configure.Vectorizer.none(),  
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="context_id", data_type=DataType.TEXT),
            ]
        )
        print("Schema created successfully")

    def add_documents(self, documents: List[Document], embeddings_model=None) -> None:
        """Add documents using BERT embeddings"""
        collection = self.client.collections.get("PMQA_Bert")
       
        with collection.batch.fixed_size(batch_size=100) as batch:
            for doc in documents:
                # Generate embedding using BERT model
                vector = self.bert_embeddings.embed_query(doc.page_content)
                
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                elif isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                if not isinstance(vector, list):
                    vector = list(vector)
                    
                batch.add_object(
                    properties={
                        "content": doc.page_content,
                        "source": str(doc.metadata.get("source", "")),
                        "document_id": str(doc.metadata.get("document_id", "")),
                        "context_id": str(doc.metadata.get("context_id", "")),
                    },
                    vector=vector
                )
        
        if collection.batch.failed_objects:
            print(f"Failed to import {len(collection.batch.failed_objects)} documents")
            print("First failure:", collection.batch.failed_objects[0])                    

    def embed_doc(self, text: str):
        """Embed document using BERT model"""
        return self.bert_embeddings.embed_query(text)

    def search_documents_with_embedding(self, query_embedding: List[float], limit: int = 2) -> List[Document]:
        """Search documents using pre-computed embedding"""
        
        # Ensure query_embedding is a proper Python list
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        elif not isinstance(query_embedding, list):
            query_embedding = list(query_embedding)
        
        print(f"Query embedding type: {type(query_embedding)}, length: {len(query_embedding)}")

        # Retrieve from the vector DB
        collection = self.client.collections.get("PMQA_Bert")
    
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=MetadataQuery(score=True, distance=True),
        )

        candidates = []
        for i, result in enumerate(results.objects):
            properties = result.properties
            metadata = result.metadata

            print(f"\nDocument {i+1} Scores (BERT):")
            print(f"Score: {metadata.score}")
            print(f"Distance: {metadata.distance}")

            doc = Document(
                page_content=properties.get("content", ""),
                metadata={
                    "source": properties.get("source", ""),
                    "document_id": properties.get("document_id", ""),
                    "context_id": properties.get("context_id", ""),
                }
            )
            
            # Add retrieval scores to metadata
            doc.metadata["score"] = metadata.score
            doc.metadata["distance"] = metadata.distance
            
            candidates.append(doc)
        
        return candidates[:limit]

    def search_documents(self, query: str, limit: int = 2) -> List[Document]:
        """Search documents using BERT embeddings for query encoding"""
        
        # Get the query embedding using BERT model
        query_embedding = self.bert_embeddings.embed_query(query)
        
        # Ensure query_embedding is a proper Python list
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        elif isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Double check it's a plain Python list
        if not isinstance(query_embedding, list):
            query_embedding = list(query_embedding)
        
        print(f"Query embedding type: {type(query_embedding)}, length: {len(query_embedding)}")

        # Retrieve from the vector DB
        collection = self.client.collections.get("PMQA_Bert")
    
        # Use hybrid search with BERT embeddings
        results = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            limit=limit,
            alpha=0.65,  # Balance between vector search (0.0) and keyword search (1.0)
            return_metadata=MetadataQuery(score=True, explain_score=True, distance=True),
        )

        candidates = []
        for i, result in enumerate(results.objects):
            properties = result.properties
            metadata = result.metadata

            print(f"\nDocument {i+1} Scores (BERT):")
            print(f"Score: {metadata.score}")
            print(f"Explain Score: {metadata.explain_score}")
            print(f"Distance: {metadata.distance}")

            doc = Document(
                page_content=properties.get("content", ""),
                metadata={
                    "source": properties.get("source", ""),
                    "document_id": properties.get("document_id", ""),
                    "context_id": properties.get("context_id", ""),
                }
            )
            
            # Add retrieval scores to metadata
            doc.metadata["score"] = metadata.score
            doc.metadata["distance"] = metadata.distance
            
            candidates.append(doc)
        
        return candidates[:limit]

    def search_documents_vector_only(self, query: str, limit: int = 2) -> List[Document]:
        """Search documents using only vector similarity (no hybrid) with BERT"""
        
        # Get the query embedding using BERT model
        query_embedding = self.bert_embeddings.embed_query(query)
        
        # Ensure query_embedding is a proper Python list
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        elif not isinstance(query_embedding, list):
            query_embedding = list(query_embedding)

        # Retrieve from the vector DB using only vector search
        collection = self.client.collections.get("PMQA_Bert")
    
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=MetadataQuery(score=True, distance=True),
        )

        candidates = []
        for i, result in enumerate(results.objects):
            properties = result.properties
            metadata = result.metadata

            doc = Document(
                page_content=properties.get("content", ""),
                metadata={
                    "source": properties.get("source", ""),
                    "document_id": properties.get("document_id", ""),
                    "context_id": properties.get("context_id", ""),
                }
            )
            
            # Add retrieval scores to metadata
            doc.metadata["score"] = metadata.score
            doc.metadata["distance"] = metadata.distance
            
            candidates.append(doc)
        
        return candidates[:limit]
    
    def get_embedding_stats(self):
        """Get statistics about the embedding collection"""
        collection = self.client.collections.get("PMQA_Bert")
        
        # Get collection info
        config = collection.config.get()
        print(f"Collection name: {config.name}")
        print(f"Vector index config: {config.vector_index_config}")
        print(f"Properties: {[prop.name for prop in config.properties]}")
        print(f"Embedding model: BERT ({self.bert_model_name})")
        
        # Get some sample embeddings to check dimensions
        results = collection.query.fetch_objects(limit=1)
        if results.objects:
            sample_vector = results.objects[0].vector
            if sample_vector:
                print(f"Embedding dimension: {len(sample_vector)}")
            else:
                print("No vector found in sample object")
        
        return config
    
    def close(self):
        self.client.close()