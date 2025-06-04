import os
import json
import glob
import pandas as pd
from typing import List, Dict, Any
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm

def load_asap_data():
    """Load ASAP data exactly like BERT training - using reviews_df with simple_decision"""
    
    # Load papers data
    papers_data = []
    review_data = []
    
    # Get all conference directories (adjust path since we're in qdrant_pipeline folder)
    dataset_path = "../dataset"
    conference_dirs = glob.glob(os.path.join(dataset_path, "*_20*"))
    
    print("Loading papers and reviews...")
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
    
    # Create DataFrames exactly like in BERT training
    papers_df = pd.DataFrame(papers_data)

    # Debug: Check if papers_df is empty or missing columns
    print(f"Papers loaded: {len(papers_df)}")
    if len(papers_df) > 0:
        print(f"Papers columns: {papers_df.columns.tolist()}")
        print(f"Sample decision: {papers_df['decision'].iloc[0] if 'decision' in papers_df.columns else 'No decision column'}")

    # Process reviews into individual rows (like BERT training)
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
                'confidence': review.get('confidence', 'N/A'),
                'review_type': 'regular'  # Default type
            })

    reviews_df = pd.DataFrame(review_rows)

    # Add simple_decision column exactly like BERT training
    if len(papers_df) > 0 and 'decision' in papers_df.columns:
        papers_df['simple_decision'] = (
            papers_df['decision']
            .str.contains('Accept', na=False)
            .map({True: 'Accept', False: 'Reject'})
        )
    else:
        print("Warning: No papers data or missing decision column")
    
    # Propagate decision to reviews_df
    reviews_df['simple_decision'] = reviews_df['paper_id'].map(
        papers_df.set_index('id')['simple_decision']
    )
    
    # Clean data exactly like BERT training
    reviews_df = reviews_df.dropna(subset=["simple_decision", "review_text"]).reset_index(drop=True)
    reviews_df["label"] = reviews_df["simple_decision"].map({"Reject": 0, "Accept": 1})
    
    print(f"Loaded {len(papers_df)} papers and {len(reviews_df)} reviews")
    print(f"Decision distribution: {reviews_df['simple_decision'].value_counts().to_dict()}")
    
    return reviews_df, papers_df

def create_rag_documents(reviews_df: pd.DataFrame) -> List[Document]:
    """Create documents for RAG system using the same data as BERT"""
    
    documents = []
    
    for idx, row in tqdm(reviews_df.iterrows(), total=len(reviews_df), desc="Creating documents"):
        # Create document content (review text is the main content for similarity)
        content = row['review_text']

        # Create metadata for retrieval and decision prediction
        metadata = {
            'paper_id': str(row['paper_id']),
            'review_idx': int(row['review_idx']),
            'rating': str(row['rating']),
            'confidence': str(row['confidence']),
            'simple_decision': str(row['simple_decision']),
            'label': int(row['label']),
            'review_text': str(row['review_text'])
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

def create_vector_database():
    """Create and populate the vector database with ASAP dataset for decision prediction"""
    
    # Load data exactly like BERT training
    reviews_df, papers_df = load_asap_data()
    
    # Create documents for RAG
    documents = create_rag_documents(reviews_df)
    
    # Split documents into chunks (smaller chunks for better retrieval)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Slightly larger to keep review context
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    
    # Load the embedding model 
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Create vector database
    url = "http://localhost:6333"
    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name="asap_decision_prediction"
    )
    
    print("ASAP Decision Prediction Vector DB Successfully Created!")
    print(f"Total documents processed: {len(documents)}")
    print(f"Total chunks created: {len(texts)}")
    
    return qdrant, reviews_df, papers_df

if __name__ == "__main__":
    create_vector_database()
