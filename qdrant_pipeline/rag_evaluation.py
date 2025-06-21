import pandas as pd
import numpy as np
import json
import glob
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import sys
sys.path.append('qdrant_pipeline')
from rag_predictor import DecisionPredictionRAG

def load_asap_data_for_evaluation():
    """Load ASAP data"""
    
    # Load papers and reviews data
    papers_data = []
    review_data = []
    
    dataset_path = "dataset"
    conference_dirs = glob.glob(os.path.join(dataset_path, "*_20*"))

    print(f"Dataset path: {dataset_path}")
    print(f"Found conference directories: {conference_dirs}")

    print("Loading papers and reviews for RAG evaluation...")
    for conf_dir in tqdm(conference_dirs):
        conf_name = os.path.basename(conf_dir)
        
        # Load papers
        paper_dir = os.path.join(conf_dir, f"{conf_name}_paper")
        if os.path.exists(paper_dir):
            paper_files = glob.glob(os.path.join(paper_dir, "*.json"))
            for paper_file in paper_files:
                try:
                    with open(paper_file, 'r', encoding='utf-8') as f:
                        paper = json.load(f)
                        papers_data.append(paper)
                except Exception as e:
                    print(f"Error loading {paper_file}: {e}")
        
        # Load reviews
        review_dir = os.path.join(conf_dir, f"{conf_name}_review")
        if os.path.exists(review_dir):
            review_files = glob.glob(os.path.join(review_dir, "*.json"))
            for review_file in review_files:
                try:
                    with open(review_file, 'r', encoding='utf-8') as f:
                        review = json.load(f)
                        review_data.append(review)
                except Exception as e:
                    print(f"Error loading {review_file}: {e}")
    
    # Create DataFrames
    papers_df = pd.DataFrame(papers_data)
    
    # Process reviews into individual rows
    review_rows = []
    for review_entry in review_data:
        paper_id = review_entry.get('id', 'N/A')
        reviews = review_entry.get('reviews', [])
        
        for i, review in enumerate(reviews):
            review_rows.append({
                'paper_id': paper_id,
                'review_idx': i,
                'review_text': review.get('review', ''),
                'rating': review.get('rating', 'N/A'),
                'confidence': review.get('confidence', 'N/A')
            })
    
    reviews_df = pd.DataFrame(review_rows)
    
    # Adding simple_decision column 
    if 'decision' in papers_df.columns:
        papers_df['simple_decision'] = (
            papers_df['decision']
            .str.contains('Accept', na=False)
            .map({True: 'Accept', False: 'Reject'})
        )
    
    # Propagate decision to reviews
    if len(reviews_df) > 0 and 'paper_id' in reviews_df.columns:
        reviews_df['simple_decision'] = reviews_df['paper_id'].map(
            papers_df.set_index('id')['simple_decision']
        )
    else:
        print(f"Warning: reviews_df columns: {reviews_df.columns.tolist()}")
        print(f"Warning: reviews_df shape: {reviews_df.shape}")
        return pd.DataFrame()  # Return empty DataFrame if structure is wrong
    
    # Clean data 
    reviews_df = reviews_df.dropna(subset=["simple_decision", "review_text"]).reset_index(drop=True)
    reviews_df["label"] = reviews_df["simple_decision"].map({"Reject": 0, "Accept": 1})
    
    print(f"Loaded {len(papers_df)} papers and {len(reviews_df)} reviews")
    print(f"Decision distribution: {reviews_df['simple_decision'].value_counts().to_dict()}")
    
    return reviews_df

def evaluate_rag_system(test_size=0.2, random_state=42, max_test_samples=200):
    """
    Evaluate RAG system using the same methodology as BERT evaluation
    Returns metrics in the same format as BERT eval_results
    """
    
    print("Starting RAG System Evaluation...")
    print("=" * 50)
    
    # Load data
    reviews_df = load_asap_data_for_evaluation()
    
    # train/test split (same as BERT)
    X = reviews_df['review_text'].tolist()
    y = reviews_df['label'].tolist()
    y_text = reviews_df['simple_decision'].tolist()
    
    # Splitting for evaluation
    X_train, X_test, y_train, y_test, y_train_text, y_test_text = train_test_split(
        X, y, y_text, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Test set: {len(X_test)} samples")
    print(f"Test distribution: {pd.Series(y_test_text).value_counts().to_dict()}")
    
    # Limit test samples for faster evaluation 
    if len(X_test) > max_test_samples:
        X_test = X_test[:max_test_samples]
        y_test = y_test[:max_test_samples]
        y_test_text = y_test_text[:max_test_samples]
        print(f"Limited to {max_test_samples} test samples for evaluation")
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = DecisionPredictionRAG()
    
    # Make predictions
    print("Making RAG predictions...")
    start_time = time.time()
    
    predictions = []
    predictions_text = []
    
    for i, review_text in enumerate(tqdm(X_test)):
        try:
            result = rag.predict_decision(review_text, return_explanation=False)
            pred_text = result['prediction']  # 'Accept' or 'Reject'
            pred_binary = 1 if pred_text == 'Accept' else 0
            
            predictions.append(pred_binary)
            predictions_text.append(pred_text)
            
        except Exception as e:
            print(f"Error predicting sample {i}: {e}")
            # Default to reject if error
            predictions.append(0)
            predictions_text.append('Reject')
    
    inference_time = time.time() - start_time
    
    # Calculate metrics 
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')
    
    # Jaccard similarity 
    jaccard = (np.array(predictions) == np.array(y_test)).mean()
    
    # Create results
    rag_eval_results = {
        'eval_accuracy': accuracy,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
        'eval_samples': len(X_test),
        'inference_time': inference_time
    }
    
    print("\nRAG System Evaluation Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Jaccard:   {jaccard:.4f}")
    print(f"Inference time: {inference_time:.2f} seconds")
    
    return rag_eval_results, jaccard, predictions, y_test

def create_comparison_table(base_eval_results, eval_results, jaccard_base, mean_jaccard):
    """
    Create comparison table including RAG system
    Uses the same format as the existing BERT comparison
    """
    
    print("\nEvaluating RAG system for comparison...")
    rag_eval_results, jaccard_rag, rag_predictions, y_test = evaluate_rag_system()
    
    # Create rows exactly like the existing comparison
    base_row = {
        'Model':    'Untrained BERT',
        'Accuracy': base_eval_results['eval_accuracy'],
        'Precision':base_eval_results['eval_precision'],
        'Recall':   base_eval_results['eval_recall'],
        'F1':       base_eval_results['eval_f1'],
        'Jaccard':  jaccard_base
    }

    fine_row = {
        'Model':    'Fine-tuned BERT',
        'Accuracy': eval_results['eval_accuracy'],
        'Precision':eval_results['eval_precision'],
        'Recall':   eval_results['eval_recall'],
        'F1':       eval_results['eval_f1'],
        'Jaccard':  mean_jaccard
    }
    
    # RAG row
    rag_row = {
        'Model':    'RAG System',
        'Accuracy': rag_eval_results['eval_accuracy'],
        'Precision':rag_eval_results['eval_precision'],
        'Recall':   rag_eval_results['eval_recall'],
        'F1':       rag_eval_results['eval_f1'],
        'Jaccard':  jaccard_rag
    }

    # Comparison DataFrame
    df_compare = pd.DataFrame([base_row, fine_row, rag_row]).set_index('Model')
    
    print("\nFinal Comparison Results:")
    print("=" * 50)
    print(df_compare.round(4))
    
    return df_compare, rag_eval_results

if __name__ == "__main__":
    # Test RAG evaluation
    rag_eval_results, jaccard_rag, predictions, y_test = evaluate_rag_system()
    
    print(f"\nRAG System Performance Summary:")
    print(f"Accuracy: {rag_eval_results['eval_accuracy']:.4f}")
    print(f"F1-Score: {rag_eval_results['eval_f1']:.4f}")
    print(f"Jaccard:  {jaccard_rag:.4f}")
