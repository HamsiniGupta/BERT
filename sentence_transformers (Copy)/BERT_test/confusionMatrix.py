#!/usr/bin/env python3
"""
Evaluate raw BERT baseline
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModel, AutoTokenizer

class RawBERTEmbeddings:
    """Raw BERT embeddings"""
    
    def __init__(self, model_name='bert-base-uncased'):
        print(f"Loading BERT: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def encode(self, texts, max_length=512):
        """Get BERT embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get BERT outputs
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
        
        return np.array(embeddings)

def evaluate_raw_bert_baseline(eval_file):
    """Evaluate BERT"""

    print("Loading BERT...")
    model = RawBERTEmbeddings('bert-base-uncased')
    
    print("Loading evaluation data...")
    df = pd.read_csv(eval_file)
    print(f"Loaded {len(df)} evaluation pairs")
    
    similarities = []
    labels = []
    
    print("Computing embeddings and similarities...")
    
    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"Processing pair {idx+1}/{len(df)}")
            
        # Get BERT embeddings
        emb1 = model.encode([row['sent1']])
        emb2 = model.encode([row['sent2']])
        
        # Convert to proper shape
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        similarities.append(similarity)
        labels.append(row['label'])
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("BERT Evaluation Results")
    print("="*50)
    
    # Basic statistics
    pos_similarities = similarities[labels == 1]
    neg_similarities = similarities[labels == 0]
    
    print(f"\nSimilarity Statistics:")
    print(f"Relevant pairs (label=1): mean={pos_similarities.mean():.4f}, std={pos_similarities.std():.4f}")
    print(f"Irrelevant pairs (label=0): mean={neg_similarities.mean():.4f}, std={neg_similarities.std():.4f}")
    print(f"Difference in means: {pos_similarities.mean() - neg_similarities.mean():.4f}")
    
    auc_score = roc_auc_score(labels, similarities)
    print(f"\nROC-AUC Score: {auc_score:.4f}")
    print("(1.0 = perfect, 0.5 = random)")
    
    # Try different thresholds for binary classification
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    best_f1 = 0
    best_threshold = 0
    
    print(f"\nThreshold Analysis:")
    print("Threshold | Accuracy | Precision | Recall | F1-Score")
    print("-" * 55)
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
        
        print(f"{threshold:8.2f} | {accuracy:8.4f} | {precision:9.4f} | {recall:6.4f} | {f1:8.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold} (F1={best_f1:.4f})")
    
    # Correlation analysis
    correlation = np.corrcoef(similarities, labels)[0, 1]
    print(f"Pearson correlation: {correlation:.4f}")
    
    # Create BERT confusion matrix
    create_raw_bert_confusion_matrix(similarities, labels, best_threshold)
    
    # Print similarity distribution analysis
    print(f"\nSimilarity Distribution Analysis:")
    print(f"Similarities below 0.3: {sum(similarities < 0.3)}")
    print(f"Similarities 0.3-0.5: {sum((similarities >= 0.3) & (similarities < 0.5))}")
    print(f"Similarities 0.5-0.7: {sum((similarities >= 0.5) & (similarities < 0.7))}")
    print(f"Similarities 0.7-0.85: {sum((similarities >= 0.7) & (similarities < 0.85))}")
    print(f"Similarities above 0.85: {sum(similarities >= 0.85)}")
    print(f"Actual range: {similarities.min():.3f} to {similarities.max():.3f}")
    print(f"Median similarity: {np.median(similarities):.3f}")
    
    # Summary assessment
    print(f"\n" + "="*50)
    print("Summary Assessment for BERT")
    print("="*50)
    
    if auc_score > 0.8:
        print("Excellent: BERT shows strong semantic understanding")
    elif auc_score > 0.7:
        print("Good: BERT captures semantic similarity well")
    elif auc_score > 0.6:
        print("Fair: BERT shows some semantic understanding")
    else:
        print("Poor: BERT needs significant improvement")
    
    print(f"Key metrics:")
    print(f"ROC-AUC: {auc_score:.4f}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Relevant vs Irrelevant gap: {pos_similarities.mean() - neg_similarities.mean():.4f}")
    
    
    # Calculate BERT metrics at threshold 0.6 for comparison
    bert_predictions_06 = (similarities > 0.6).astype(int)
    bert_acc_06 = accuracy_score(labels, bert_predictions_06)
    bert_prec_06, bert_rec_06, bert_f1_06, _ = precision_recall_fscore_support(labels, bert_predictions_06, average='binary', zero_division=0)
    
    # Also at best threshold
    bert_predictions_best = (similarities > best_threshold).astype(int)
    bert_acc_best = accuracy_score(labels, bert_predictions_best)
    bert_prec_best, bert_rec_best, bert_f1_best, _ = precision_recall_fscore_support(labels, bert_predictions_best, average='binary', zero_division=0)
    
    print(f"\nBERT results")
    print(f"Accuracy: {bert_acc_06:.1%}")
    print(f"Precision: {bert_prec_06:.1%}")
    print(f"Recall: {bert_rec_06:.1%}")
    print(f"F1: {bert_f1_06:.1%}")
    print(f"ROC-AUC: {auc_score:.3f}")
    
    print(f"\nBERT results at best threshold {best_threshold}:")
    print(f"Accuracy: {bert_acc_best:.1%}")
    print(f"Precision: {bert_prec_best:.1%}")
    print(f"Recall: {bert_rec_best:.1%}")
    print(f"F1: {bert_f1_best:.1%}")
    
    return {
        'similarities': similarities,
        'labels': labels,
        'auc_score': auc_score,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'accuracy_at_06': bert_acc_06,
        'f1_at_06': bert_f1_06
    }

def create_raw_bert_confusion_matrix(similarities, labels, threshold, save_path="../plots/raw_bert_confusion_matrix.png"):
    """Create confusion matrix for BERT"""
    
    # Create predictions
    predictions = (similarities > threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    
    # Create horizontal figure
    plt.figure(figsize=(10, 6))
    
    # Create heatmap with same green colormap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Irrelevant', 'Relevant'],
                yticklabels=['Irrelevant', 'Relevant'],
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 22})
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title(f'Confusion Matrix for Raw BERT', fontsize=16, fontweight='bold')
    
    # Print confusion matrix values
    print(f"\nBERT Confusion Matrix:")
    print(f"True Negatives (TN): {cm[0,0]}")
    print(f"False Positives (FP): {cm[0,1]}")
    print(f"False Negatives (FN): {cm[1,0]}")
    print(f"True Positives (TP): {cm[1,1]}")
    
    # Calculate metrics
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nBERT Performance (at threshold {threshold}):")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1-Score: {f1:.1%}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"BERT confusion matrix saved to {save_path}")
    
    return cm

if __name__ == "__main__":
    
    import os

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    EVAL_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", "pubmedqa_test_clean.csv"))

    print("Evaluating BERT...")
    
    results = evaluate_raw_bert_baseline(EVAL_DATA_PATH)
    
    print("\nEvaluation complete!")
    print("Check plots/raw_bert_confusion_matrix.png for BERT.")
