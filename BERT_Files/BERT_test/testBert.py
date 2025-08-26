#!/usr/bin/env python3

import pandas as pd
import time
import json
import os
from typing import List, Dict
from collections import defaultdict

def generate_test_results(pipeline, weaviate_manager, embeddings, max_samples=50, output_file="pubmedbert_test_results.csv", dataset_file="../data/actual_testing_dataset.json"):
    
    # Check if the dataset file exists
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return None
    
    print(f"Loading test dataset from {dataset_file}...")
    
    # Load your existing JSON dataset
    with open(dataset_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} items from {dataset_file}")
    
    # Convert to list of items for processing
    test_items = []
    for pmid, item in raw_data.items():
        # Extract the relevant fields
        question = item.get("QUESTION", "").strip()
        final_decision = item.get("final_decision", "").lower().strip()
        
        # Handle different answer formats
        if final_decision not in ["yes", "no", "maybe"]:
            # Try alternative fields if final_decision is not standard
            alt_decision = item.get("reasoning_required_pred", "").lower().strip()
            if alt_decision in ["yes", "no", "maybe"]:
                final_decision = alt_decision
            else:
                alt_decision = item.get("reasoning_free_pred", "").lower().strip()
                if alt_decision in ["yes", "no", "maybe"]:
                    final_decision = alt_decision
                else:
                    print(f"Warning: No valid decision found for {pmid}, skipping...")
                    continue
        
        if not question:
            print(f"Warning: Empty question for {pmid}, skipping...")
            continue
        
        test_items.append({
            'question': question,
            'final_decision': final_decision,
            'long_answer': item.get("LONG_ANSWER", ""),
            'pubid': pmid,
            'contexts': item.get("CONTEXTS", []),
            'year': item.get("YEAR", ""),
            'meshes': item.get("MESHES", [])
        })
    
    print(f"Processed {len(test_items)} items for testing")
    
    # Limit the number of samples if specified
    if max_samples and max_samples < len(test_items):
        test_items = test_items[:max_samples]
        print(f"Limited to {max_samples} samples for testing")
    
    results = []
    
    print(f"Generating results for {len(test_items)} questions using BERT...")
    
    for i, item in enumerate(test_items):
        question = item['question']
        true_answer = item['final_decision']
        long_answer = item['long_answer']
        pubid = item['pubid']
        
        print(f"Processing question {i+1}/{len(test_items)}: {question[:50]}...")
        
        try:
            start_time = time.time()
    
            print(f"Searching with BERT for: {question[:50]}...")
            
            retrieved_docs = weaviate_manager.search_documents(query=question, limit=2)
            
            context1 = retrieved_docs[0].page_content if len(retrieved_docs) > 0 else "No context retrieved"
            context2 = retrieved_docs[1].page_content if len(retrieved_docs) > 1 else "No second context"
            
            # Log what was actually retrieved
            print(f"Retrieved {len(retrieved_docs)} documents")
            if retrieved_docs:
                print(f"First doc preview: {context1[:100]}...")
            
            rag_response = pipeline.invoke(question)
            print(f"BERT response: {rag_response[:100]}")
            
            elapsed_time = time.time() - start_time
            
            # Extract scores if available
            doc1_score = retrieved_docs[0].metadata.get('score', 0) if len(retrieved_docs) > 0 else 0
            doc2_score = retrieved_docs[1].metadata.get('score', 0) if len(retrieved_docs) > 1 else 0
            
            try:
                # Test embedding to get dimension
                test_embedding = weaviate_manager.bert_embeddings.embed_query("test")
                embedding_dim = len(test_embedding) if hasattr(test_embedding, '__len__') else 768
            except:
                embedding_dim = 768  # Default BERT dimension
            
            result = {
                'question': question,
                'Context1': context1,
                'Context2': context2,
                'Answer': rag_response,
                'Ground_truth': true_answer,
                'Long_answer': long_answer,
                'pubid': pubid,
                'response_time': elapsed_time,
                'retrieval_score_1': doc1_score,
                'retrieval_score_2': doc2_score,
                'question_id': i,
                'embedding_model': 'BERT', 
                'embedding_dimension': embedding_dim,
                'year': item.get('year', ''),
                'meshes': str(item.get('meshes', []))  # Convert list to string for CSV
            }
            
        except Exception as e:
            print(f"Error processing question {i+1} with BERT: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'question': question,
                'Context1': "Error retrieving context with BERT",
                'Context2': "Error retrieving context with BERT", 
                'Answer': f"BERT Error: {str(e)}",
                'Ground_truth': true_answer,
                'Long_answer': long_answer,
                'pubid': pubid,
                'response_time': 0,
                'retrieval_score_1': 0,
                'retrieval_score_2': 0,
                'question_id': i,
                'embedding_model': 'BERT',
                'embedding_dimension': 768,
                'year': item.get('year', ''),
                'meshes': str(item.get('meshes', []))
            }
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} questions with BERT...")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Generated {len(results)} test results using BERT")
    print(f"Columns: {list(df.columns)}")
   
    return df

def quick_accuracy_check(df):
    def extract_decision(text):
        if not text or pd.isna(text):
            return 'unknown'
        text = str(text).lower().strip()
        print(f"BERT response: {text}")
        if 'yes' in text:
            return 'yes'
        elif 'no' in text:
            return 'no' 
        elif 'maybe' in text:
            return 'maybe'
        return 'unknown'
    
    df['predicted_label'] = df['Answer'].apply(extract_decision)
    df['true_label'] = df['Ground_truth'].apply(extract_decision)
    
    correct = (df['predicted_label'] == df['true_label']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nQuick Accuracy Check:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.3f}")
    
    return accuracy

def calculate_precision_recall_f1(df):
    """
    Calculate precision, recall, and F1 score for each class
    """
    
    def extract_decision(text):
        if not text or pd.isna(text):
            return 'unknown'
        text = str(text).lower().strip()
        if 'yes' in text:
            return 'yes'
        elif 'no' in text:
            return 'no' 
        elif 'maybe' in text:
            return 'maybe'
        return 'unknown'
    
    # Apply extraction if not already done
    if 'predicted_label' not in df.columns:
        df['predicted_label'] = df['Answer'].apply(extract_decision)
    if 'true_label' not in df.columns:
        df['true_label'] = df['Ground_truth'].apply(extract_decision)
    
    # Get unique labels
    all_labels = sorted(set(df['true_label'].tolist() + df['predicted_label'].tolist()))
    
    # Calculate metrics for each class
    metrics = {}
    
    for label in all_labels:
        # True Positives: predicted as label AND actually label
        tp = len(df[(df['predicted_label'] == label) & (df['true_label'] == label)])
        
        # False Positives: predicted as label BUT actually not label
        fp = len(df[(df['predicted_label'] == label) & (df['true_label'] != label)])
        
        # False Negatives: predicted as not label BUT actually label
        fn = len(df[(df['predicted_label'] != label) & (df['true_label'] == label)])
        
        # True Negatives: predicted as not label AND actually not label
        tn = len(df[(df['predicted_label'] != label) & (df['true_label'] != label)])
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    # Calculate overall accuracy
    correct = (df['predicted_label'] == df['true_label']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0
    
    print("\n" + "="*60)
    print("Precision, Recall, and F1 Score")
    print("="*60)
    
    # Header
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 48)
    
    class_order = ['maybe', 'no', 'yes']
    ordered_labels = [label for label in class_order if label in all_labels]
    ordered_labels.extend([label for label in all_labels if label not in class_order])
    
    for label in ordered_labels:
        if label in metrics:
            precision = metrics[label]['precision']
            recall = metrics[label]['recall']
            f1 = metrics[label]['f1']
            
            print(f"{label.upper():<10} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}")
    
    print("-" * 48)
    print(f"{'Accuracy':<10} {'':<12} {'':<12} {accuracy:<12.3f}")
    
    print("\n" + "="*60)
    print("Metrics Table")
    print("="*60)
    
    for label in ordered_labels:
        if label in metrics:
            precision = metrics[label]['precision']
            recall = metrics[label]['recall']
            f1 = metrics[label]['f1']
            
            print(f"{label.upper():<10} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")
    
    print(f"{'Accuracy':<10} {'':<10} {'':<10} {accuracy:>10.3f}")
    
    return metrics, accuracy

def detailed_confusion_analysis(df):
    """
    Detailed analysis of model predictions vs ground truth
    Shows confusion matrix and specific error patterns
    """
    
    def extract_decision(text):
        if not text or pd.isna(text):
            return 'unknown'
        text = str(text).lower().strip()
        if 'yes' in text:
            return 'yes'
        elif 'no' in text:
            return 'no' 
        elif 'maybe' in text:
            return 'maybe'
        return 'unknown'
    
    # Apply extraction if not already done
    if 'predicted_label' not in df.columns:
        df['predicted_label'] = df['Answer'].apply(extract_decision)
    if 'true_label' not in df.columns:
        df['true_label'] = df['Ground_truth'].apply(extract_decision)
    
    print("="*60)
    print("Confusion Matrix")
    print("="*60)
    
    # Get unique labels
    all_labels = sorted(set(df['true_label'].tolist() + df['predicted_label'].tolist()))
    
    # Create confusion matrix 
    confusion_counts = {}
    for true_label in all_labels:
        for pred_label in all_labels:
            count = len(df[(df['true_label'] == true_label) & (df['predicted_label'] == pred_label)])
            confusion_counts[(true_label, pred_label)] = count
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("PREDICTED")
    print(" ", end="")
    for pred_label in all_labels:
        print(f"{pred_label:>8}", end="")
    print()
    
    for true_label in all_labels:
        print(f"TRUE {true_label:>6}", end="")
        for pred_label in all_labels:
            count = confusion_counts[(true_label, pred_label)]
            print(f"{count:>8}", end="")
        print()
    
    # Detailed breakdown
    print(f"\nError Analysis")
    total_questions = len(df)
    
    for true_label in all_labels:
        for pred_label in all_labels:
            count = confusion_counts[(true_label, pred_label)]
            if count > 0:
                percentage = (count / total_questions) * 100
                if true_label == pred_label:
                    print(f"TRUE '{true_label}' PREDICTED '{pred_label}': {count} cases ({percentage:.1f}%) - CORRECT")
                else:
                    print(f"TRUE '{true_label}' PREDICTED '{pred_label}': {count} cases ({percentage:.1f}%) - ERROR")
    
    # Calculate class accuracy
    print(f"\nClass Performance")
    for true_label in all_labels:
        total_true = sum(confusion_counts[(true_label, pred)] for pred in all_labels)
        correct_true = confusion_counts[(true_label, true_label)]
        if total_true > 0:
            accuracy = (correct_true / total_true) * 100
            print(f"  {true_label.upper()}: {correct_true}/{total_true} correct ({accuracy:.1f}%)")
    
    # Most common errors
    print(f"\nMost Common Errors")
    errors = [(true_label, pred_label, count) for (true_label, pred_label), count in confusion_counts.items() 
              if true_label != pred_label and count > 0]
    errors.sort(key=lambda x: x[2], reverse=True)
    
    for i, (true_label, pred_label, count) in enumerate(errors[:5]):
        percentage = (count / total_questions) * 100
        print(f"  {i+1}. Should be '{true_label}' but predicted '{pred_label}': {count} times ({percentage:.1f}%)")
    
    # Show example errors for the most common mistake
    if errors:
        most_common_error = errors[0]
        true_label, pred_label, count = most_common_error
        
        print(f"\nExample cases where TRUE='{true_label}' but PREDICTED='{pred_label}' (BERT):")
        error_cases = df[(df['true_label'] == true_label) & (df['predicted_label'] == pred_label)]
        
        for i, (_, row) in enumerate(error_cases.head(3).iterrows()):
            print(f"\nExample {i+1}:")
            print(f"Question: {row['question'][:80]}...")
            print(f"Ground Truth: {row['Ground_truth']}")
            print(f"Model Answer: {row['Answer'][:100]}...")
            print(f"Context Quality Score: {row.get('retrieval_score_1', 'N/A')}")
            print(f"Embedding Model: {row.get('embedding_model', 'Unknown')}")
    
    return confusion_counts

def run_test_generation(pipeline, weaviate_manager, embeddings, dataset_file="../data/actual_testing_dataset.json"):
    
    # Generate test results using dataset
    output_file = "../data/pubmedbert_evaluation_data.csv"
    
    print("Starting test result generation for BERT...")
    print(f"Using dataset: {dataset_file}")
    
    df = generate_test_results(
        pipeline=pipeline,
        weaviate_manager=weaviate_manager, 
        embeddings=embeddings,
        max_samples=None,  # Use all samples 
        output_file=output_file,
        dataset_file=dataset_file  # Pass dataset file
    )
    
    if df is None:
        print("Failed to generate results")
        return None, None, None
    
    accuracy = quick_accuracy_check(df)
    
    metrics, overall_accuracy = calculate_precision_recall_f1(df)
    
    #confusion analysis
    confusion_matrix = detailed_confusion_analysis(df)
    print(f"{confusion_matrix}")
    
    print(f"\nGenerated BERT evaluation dataset:")
    print(f"File: {output_file}")
    print(f"Questions: {len(df)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Embedding Model: BERT")
    print(f"Source Dataset: {dataset_file}")
    # print(f"Use this file in:")
    # print(f"AnswerRelevancy.ipynb")
    # print(f"ContextAdherence.ipynb") 
    # print(f"ContextRelevancy.ipynb")
    
    return df, output_file, metrics

if __name__ == "__main__":
    print("Import this script and call run_test_generation() with pipeline objects")
    print("Usage: run_test_generation(pipeline, weaviate_manager, embeddings, dataset_file='../data/actual_testing_dataset.json')")