import pandas as pd
import os
import re
from typing import List, Dict
import torch
import time
import numpy as np

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline
from langchain_huggingface.llms import HuggingFacePipeline

# =====================

from huggingface_hub import HfApi
from PubMedQAProcessor import PubMedQAProcessor
from WeaviateManager import WeaviateManager
from RAGPipeline import RAGPipeline

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print(f"HF Token loaded: {bool(os.getenv('HUGGINGFACE_TOKEN'))}")
print(f"Weaviate key loaded: {bool(os.getenv('WEAVIATE_API_KEY'))}")

# Setting HF token
# Access the token from the environment variable

hf_token = os.getenv('HUGGINGFACE_TOKEN')
if not hf_token:
    raise ValueError("HuggingFace token not found in environment variables")

# Use the token to authenticate
api = HfApi()

# =====================
from transformers import AutoModelForCausalLM, AutoTokenizer

# Defining LLM 
def llm_model(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    llm.model_rebuild()  # Fixes the PydanticUserError
    return llm

# =====================

BERT_MODEL_NAME = "google-bert/bert-base-uncased"


from langchain_core.embeddings import Embeddings

class BERTEmbeddingsWrapper(Embeddings):
    """Wrapper BERT Embeddings compatible with LangChain"""
    
    def __init__(self, weaviate_bert_embeddings):
        self.bert_embeddings = weaviate_bert_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings_array = self.bert_embeddings.encode(texts)
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings_array]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.bert_embeddings.embed_query(text)
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        elif isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return list(embedding)

WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
if not WEAVIATE_API_KEY:
    raise ValueError("Weaviate API key not found in environment variables")
if not WEAVIATE_URL:
    raise ValueError("Weaviate URL not found in environment variables")

COLLECTION_NAME = "PMQA_Bert" 

def initialize_rag_with_bert():
    """Initialize the RAG system with BERT model"""
    print("Initializing LLM...")
    llm = llm_model("meta-llama/Meta-Llama-3.1-8B-Instruct")

    print("Connecting to Weaviate with BERT...")
    weaviate_manager = WeaviateManager(WEAVIATE_URL, WEAVIATE_API_KEY, hf_token, BERT_MODEL_NAME)

    print("Loading and processing PubMedQA dataset...")
    dataset = PubMedQAProcessor.load_pubmedqa_dataset()
    documents = PubMedQAProcessor.process_contexts_to_documents(dataset)

    print("Reindexing documents with BERT...")
    collection_name = weaviate_manager.reindex_with_embeddings(documents, weaviate_manager.bert_embeddings)
    
    weaviate_manager.collection_name = collection_name
    print(f"Using collection: {collection_name}")

    print("Creating RAG pipeline...")
    # Create wrapper for LangChain compatibility
    embeddings_wrapper = BERTEmbeddingsWrapper(weaviate_manager.bert_embeddings)
    
    rag = RAGPipeline(weaviate_manager, embeddings_wrapper, llm)
    pipeline = rag.create_pipeline()

    return pipeline, weaviate_manager, embeddings_wrapper

# =====================

# Load dataset directly if embeddings already stored
def load_initialized_rag():
    """Load an already initialized RAG system without reloading the dataset."""
    print("Initializing LLM...")
    llm = llm_model("meta-llama/Meta-Llama-3.1-8B-Instruct")

    print("Connecting to Weaviate with BERT...")
    weaviate_manager = WeaviateManager(WEAVIATE_URL, WEAVIATE_API_KEY, hf_token, BERT_MODEL_NAME)
    
    weaviate_manager.collection_name = COLLECTION_NAME
    
    print("Creating RAG pipeline...")
    # Create wrapper for LangChain compatibility
    embeddings_wrapper = BERTEmbeddingsWrapper(weaviate_manager.bert_embeddings)
    
    rag = RAGPipeline(weaviate_manager, embeddings_wrapper, llm)
    pipeline = rag.create_pipeline()

    return pipeline, weaviate_manager, embeddings_wrapper  

FORCE_REINDEX = True

# Temporary client to check collection existence
temp_manager = WeaviateManager(WEAVIATE_URL, WEAVIATE_API_KEY, hf_token, BERT_MODEL_NAME)
client = temp_manager.client

print(f"Checking for collection: {COLLECTION_NAME}")

if FORCE_REINDEX:
    print("Deleting existing collection...")
    client.collections.delete(COLLECTION_NAME)
    pipeline, weaviate_manager, embeddings = initialize_rag_with_bert()

elif client.collections.exists(COLLECTION_NAME):
    print("Collection exists. Loading RAG...")
    pipeline, weaviate_manager, embeddings = load_initialized_rag()

else:
    print("Collection does not exist. Initializing RAG...")
    pipeline, weaviate_manager, embeddings = initialize_rag_with_bert()

import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams

# Import the test generation script
from testBert import run_test_generation

print("\nGenerating Test Results for Evaluation...")

# Generate test results using loaded pipeline
df, results_file = run_test_generation(
    pipeline=pipeline,
    weaviate_manager=weaviate_manager,
    embeddings=embeddings
)

# print(f"\nTest generation completed!")
# print(f"Results saved to: {results_file}")
# print(f"\nNext steps:")
# print(f"1. Use {results_file} in AnswerRelevancy.ipynb")
# print(f"2. Use {results_file} in ContextAdherence.ipynb") 
# print(f"3. Use {results_file} in ContextRelevancy.ipynb")

weaviate_manager.close()
client.close()

print("Weaviate client closed")