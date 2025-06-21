import os
import json
import glob
import pandas as pd
import nltk
from typing import List, Dict, Any
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm

# Initialize NLTK
print("=" * 50)
print("INITIALIZING NLTK")
print("=" * 50)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("SUCCESS: NLTK data downloaded successfully")
except Exception as e:
    print(f"WARNING: NLTK download warning: {e}")
print("SUCCESS: NLTK READY")
print("=" * 50)
print()

def load_asap_data():
    """Load ASAP data exactly like BERT training - using reviews_df with simple_decision"""

    print("=" * 50)
    print("LOADING ASAP DATASET")
    print("=" * 50)

    # Load papers data
    papers_data = []
    review_data = []

    # Get all conference directories (adjust path since we're in qdrant_pipeline folder)
    dataset_path = "dataset"
    conference_dirs = glob.glob(os.path.join(dataset_path, "*_20*"))

    print(f"FOUND: {len(conference_dirs)} conference directories:")
    for conf_dir in conference_dirs:
        print(f"   - {os.path.basename(conf_dir)}")
    print()

    print("STEP 1: Loading papers and reviews...")
    for conf_dir in tqdm(conference_dirs, desc="Processing conferences"):
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
    
    print("=" * 50)
    print("STEP 2: PROCESSING DATA")
    print("=" * 50)

    # Create DataFrames exactly like in BERT training
    print("Creating papers DataFrame...")
    papers_df = pd.DataFrame(papers_data)

    # Debug: Check if papers_df is empty or missing columns
    print(f"SUCCESS: Papers loaded: {len(papers_df)}")
    if len(papers_df) > 0:
        print(f"Papers columns: {papers_df.columns.tolist()}")
        print(f"Sample decision: {papers_df['decision'].iloc[0] if 'decision' in papers_df.columns else 'No decision column'}")
    print()

    # Process reviews into individual rows (like BERT training)
    print("Processing reviews into individual rows...")
    review_rows = []
    for review_entry in tqdm(review_data, desc="Processing review entries"):
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

    print("Creating reviews DataFrame...")
    reviews_df = pd.DataFrame(review_rows)

    print("STEP 3: Adding decision labels...")
    # Add simple_decision column exactly like BERT training
    if len(papers_df) > 0 and 'decision' in papers_df.columns:
        papers_df['simple_decision'] = (
            papers_df['decision']
            .str.contains('Accept', na=False)
            .map({True: 'Accept', False: 'Reject'})
        )
        print("SUCCESS: Simple decision labels created")
    else:
        print("ERROR: No papers data or missing decision column")
        return pd.DataFrame(), pd.DataFrame()

    print("STEP 4: Propagating decisions to reviews...")
    # Propagate decision to reviews_df
    reviews_df['simple_decision'] = reviews_df['paper_id'].map(
        papers_df.set_index('id')['simple_decision']
    )

    print("STEP 5: Cleaning data...")
    # Clean data exactly like BERT training
    initial_count = len(reviews_df)
    reviews_df = reviews_df.dropna(subset=["simple_decision", "review_text"]).reset_index(drop=True)
    reviews_df["label"] = reviews_df["simple_decision"].map({"Reject": 0, "Accept": 1})
    final_count = len(reviews_df)

    print("=" * 50)
    print("SUCCESS: DATA LOADING COMPLETE")
    print("=" * 50)
    print(f"Papers loaded: {len(papers_df)}")
    print(f"Reviews loaded: {final_count} (cleaned from {initial_count})")
    print(f"Decision distribution: {reviews_df['simple_decision'].value_counts().to_dict()}")
    print("=" * 50)
    print()

    return reviews_df, papers_df

def create_rag_documents(reviews_df: pd.DataFrame) -> List[Document]:
    """Create documents for RAG system using the same data as BERT"""

    print("=" * 50)
    print("STEP 6: CREATING RAG DOCUMENTS")
    print("=" * 50)

    documents = []

    for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df), desc="Creating documents"):
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

    print(f"SUCCESS: Created {len(documents)} documents")
    print("=" * 50)
    print()

    return documents

def create_vector_database():
    """Create and populate the vector database with ASAP dataset for decision prediction"""

    print("=" * 50)
    print("STARTING VECTOR DATABASE CREATION")
    print("=" * 50)

    # Load data exactly like BERT training
    reviews_df, papers_df = load_asap_data()

    # Check if data loading was successful
    if len(reviews_df) == 0:
        print("ERROR: No review data loaded. Exiting.")
        return None, None, None

    # Create documents for RAG
    documents = create_rag_documents(reviews_df)

    print("STEP 7: Splitting documents into chunks...")
    # Split documents into chunks (smaller chunks for better retrieval)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Slightly larger to keep review context
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"SUCCESS: Created {len(texts)} text chunks from {len(documents)} documents")
    print()

    print("STEP 8: Loading embedding model...")
    # Load the embedding model
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    print(f"Model: {model_name}")
    print("Device: CPU")

    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("SUCCESS: Embedding model loaded")
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        return None, None, None
    print()

    print("STEP 9: Creating vector database...")
    # Create vector database
    url = "http://localhost:6333"
    print(f"Connecting to Qdrant at: {url}")
    print(f"Collection name: asap_decision_prediction")

    try:
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=url,
            prefer_grpc=False,
            collection_name="asap_decision_prediction"
        )
        print("SUCCESS: Vector database created!")
    except Exception as e:
        print(f"ERROR: Failed to create vector database: {e}")
        return None, None, None

    print("=" * 50)
    print("SUCCESS: VECTOR DATABASE CREATION COMPLETE")
    print("=" * 50)
    print(f"Total documents processed: {len(documents)}")
    print(f"Total chunks created: {len(texts)}")
    print(f"Collection: asap_decision_prediction")
    print(f"Qdrant URL: {url}")
    print("=" * 50)

    return qdrant, reviews_df, papers_df

if __name__ == "__main__":
    create_vector_database()
