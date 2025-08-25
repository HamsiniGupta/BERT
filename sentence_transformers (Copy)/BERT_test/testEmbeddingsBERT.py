#!/usr/bin/env python3
"""
Evaluate baseline BERT model on pairs for comparison with PubMedBERT
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json

class BERTEmbeddings:
    def __init__(self, model_name="google-bert/bert-base-uncased"):
        """Initialize BERT model for embeddings"""
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def embed_query(self, text):
        """Get BERT embedding for a text"""
        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, 
                               truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]  # Return 1D array

def evaluate_bert_baseline(model_name="google-bert/bert-base-uncased", eval_file="data/pubmedqa_val_clean.csv"):
    """
    Evaluate baseline BERT model on the same validation set
    """
    print("="*60)
    print(f"BERT BASELINE EVALUATION: {model_name}")
    print("="*60)
    
    # Initialize BERT model
    embeddings_model = BERTEmbeddings(model_name)
    
    print("Loading evaluation data...")
    df = pd.read_csv(eval_file)
    print(f"Loaded {len(df)} evaluation pairs")
    
    similarities = []
    labels = []
    
    print("Computing embeddings and similarities...")
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"Processing pair {idx+1}/{len(df)}")
            
        # Get embeddings for both sentences
        emb1 = embeddings_model.embed_query(row['sent1'])
        emb2 = embeddings_model.embed_query(row['sent2'])
        
        # Convert to numpy arrays and reshape
        emb1 = np.array(emb1).reshape(1, -1)
        emb2 = np.array(emb2).reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        similarities.append(similarity)
        labels.append(row['label'])
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("BERT BASELINE EVALUATION RESULTS")
    print("="*50)
    
    # Basic statistics
    pos_similarities = similarities[labels == 1]
    neg_similarities = similarities[labels == 0]
    
    print(f"\nSimilarity Statistics:")
    print(f"Relevant pairs (label=1):   mean={pos_similarities.mean():.4f}, std={pos_similarities.std():.4f}")
    print(f"Irrelevant pairs (label=0): mean={neg_similarities.mean():.4f}, std={neg_similarities.std():.4f}")
    print(f"Difference in means: {pos_similarities.mean() - neg_similarities.mean():.4f}")
    
    # ROC-AUC 
    auc_score = roc_auc_score(labels, similarities)
    print(f"\nROC-AUC Score: {auc_score:.4f}")
    print("(1.0 = perfect, 0.5 = random)")
    
    # Try different thresholds for binary classification
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8]
    best_f1 = 0
    best_threshold = 0
    
    print(f"\nThreshold Analysis:")
    print("Threshold | Accuracy | Precision | Recall | F1-Score")
    print("-" * 55)
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        print(f"{threshold:8.1f} | {accuracy:8.4f} | {precision:9.4f} | {recall:6.4f} | {f1:8.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold} (F1={best_f1:.4f})")
    
    # Correlation analysis
    correlation = np.corrcoef(similarities, labels)[0, 1]
    print(f"Pearson correlation: {correlation:.4f}")
    
    # Create visualization - match PubMedBERT style
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Similarity distributions - match PubMedBERT style
    plt.subplot(1, 2, 1)
    plt.hist(neg_similarities, bins=30, alpha=0.7, label='Irrelevant (0)', color='red')
    plt.hist(pos_similarities, bins=30, alpha=0.7, label='Relevant (1)', color='green')
    plt.xlabel('Cosine Similarity', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title(f'Similarity Distribution by Label', fontsize=18)
    plt.legend(fontsize=13)
    # Set y-axis limit to match PubMedBERT format
    plt.ylim(0, 25)
    # Set x-axis limit to match PubMedBERT format (0.3 to 0.9)
    plt.xlim(0.3, 0.9)
    
    # Plot 2: Scatter plot
    plt.subplot(1, 2, 2)
    jittered_labels = labels + np.random.normal(0, 0.05, len(labels))
    plt.scatter(similarities, jittered_labels, alpha=0.6)
    plt.xlabel('Cosine Similarity', fontsize=18)
    plt.ylabel('Label (jittered)', fontsize=18)
    plt.title('Similarity vs Label', fontsize=18)
    plt.ylim(-0.5, 1.5)
    
    plt.tight_layout()
    
    # Save plot with model name
    plt.savefig(f'../plots/BERT_Embeddings_baseline_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary assessment
    print(f"\n" + "="*50)
    print("SUMMARY ASSESSMENT")
    print("="*50)
    
    if auc_score > 0.8:
        print("EXCELLENT: shows strong semantic understanding")
    elif auc_score > 0.7:
        print("GOOD: captures semantic similarity well")
    elif auc_score > 0.6:
        print("FAIR: some semantic understanding")
    else:
        print("POOR: may need more training or data")
    
    print(f"Key metrics:")
    print(f"- ROC-AUC: {auc_score:.4f}")
    print(f"- Best F1: {best_f1:.4f}")
    print(f"- Relevant vs Irrelevant gap: {pos_similarities.mean() - neg_similarities.mean():.4f}")
    
    return {
        'model_name': model_name,
        'similarities': similarities,
        'labels': labels,
        'auc_score': auc_score,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'pos_mean': pos_similarities.mean(),
        'neg_mean': neg_similarities.mean(),
        'correlation': correlation
    }

def save_results(results, filename):
    """Save evaluation results to JSON file"""
    # Convert numpy arrays and numpy scalars to Python types for JSON serialization
    results_copy = {}
    
    for key, value in results.items():
        if hasattr(value, 'tolist'):  # numpy arrays
            results_copy[key] = value.tolist()
        elif hasattr(value, 'item'):  # numpy scalars
            results_copy[key] = value.item()
        else:  # regular Python types
            results_copy[key] = value
    
    with open(filename, 'w') as f:
        json.dump(results_copy, f, indent=2)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    import os
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    EVAL_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", "pubmedqa_val_clean.csv"))
    
    # Test only BERT baseline for comparison with PubMedBERT
    model_name = "google-bert/bert-base-uncased"
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    bert_results = evaluate_bert_baseline(model_name, EVAL_DATA_PATH)
    save_results(bert_results, 'bert_baseline_results.json')
    
    print(f"\nBERT baseline evaluation complete!")
    print(f"Results saved to: ../plots/BERT_Embeddings_baseline_results.png")
    print(f"\nTo compare with PubMedBERT, run your PubMedBERT evaluation script")
    print(f"and use the compare_models() function with both results.")