import os
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DecisionPredictionRAG:
    """RAG system for predicting paper decisions based on review similarity"""
    
    def __init__(self, openai_model: str = "gpt-3.5-turbo"):
        """Initialize the RAG system"""
        self.openai_model = openai_model
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize embeddings (same as ingestion)
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Connect to Qdrant
        self.vector_db = None
        self.connect_to_qdrant()
    
    def connect_to_qdrant(self):
        """Connect to the Qdrant vector database"""
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url="http://localhost:6333")
            self.vector_db = Qdrant(
                client=client,
                embeddings=self.embeddings,
                collection_name="asap_decision_prediction"
            )
            print("Connected to Qdrant vector database")
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            print("Make sure Qdrant is running and the collection exists")
    
    def retrieve_similar_reviews(self, query_text: str, k: int = 5) -> List[Dict]:
        """Retrieve similar reviews from the vector database"""
        if not self.vector_db:
            raise ValueError("Vector database not connected")
        
        try:
            # Search for similar reviews
            docs = self.vector_db.similarity_search(query_text, k=k)
            
            similar_reviews = []
            for doc in docs:
                similar_reviews.append({
                    'review_text': doc.metadata.get('review_text', ''),
                    'decision': doc.metadata.get('simple_decision', ''),
                    'rating': doc.metadata.get('rating', ''),
                    'confidence': doc.metadata.get('confidence', ''),
                    'paper_id': doc.metadata.get('paper_id', '')
                })
            
            return similar_reviews
        except Exception as e:
            print(f"Error retrieving similar reviews: {e}")
            return []
    
    def predict_decision(self, review_text: str, return_explanation: bool = True) -> Dict[str, Any]:
        """Predict decision for a given review text"""
        
        # Retrieve similar reviews
        similar_reviews = self.retrieve_similar_reviews(review_text, k=5)
        
        if not similar_reviews:
            return {
                'prediction': 'Reject',  # Default prediction
                'confidence': 0.5,
                'explanation': 'No similar reviews found in database'
            }
        
        # Build context from similar reviews
        context = self._build_context(similar_reviews)
        
        # Create prompt for OpenAI
        prompt = self._create_prediction_prompt(review_text, context, return_explanation)
        
        try:
            # Enhanced system prompt for better explanations
            system_prompt = """You are an expert academic paper reviewer with extensive experience in evaluating research papers for top-tier conferences like ICLR, NIPS, and other ML/AI venues.

Your task is to predict whether a paper will be ACCEPTED or REJECTED based on review patterns from similar papers in the database.

When making predictions, consider these key factors:
- Technical quality and novelty of the contribution
- Clarity of writing and presentation
- Experimental rigor and completeness
- Significance of results and impact
- Comparison with existing work
- Reviewer confidence and rating patterns

Provide detailed reasoning that explains:
1. What patterns you observed in the similar reviews
2. Which specific aspects (positive/negative) influenced your decision
3. How the review language and ratings correlate with the final decision
4. Any red flags or strong positive indicators you noticed"""

            # Get prediction from OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Slightly higher for more nuanced explanations
                max_tokens=800  # More tokens for detailed explanations
            )
            
            response_text = response.choices[0].message.content
            result = self._parse_openai_response(response_text)
            result['similar_reviews'] = similar_reviews
            return result
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Fallback: majority vote from similar reviews
            decisions = [r['decision'] for r in similar_reviews if r['decision']]
            if decisions:
                prediction = max(set(decisions), key=decisions.count)
                confidence = decisions.count(prediction) / len(decisions)
            else:
                prediction = 'Reject'
                confidence = 0.5
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'explanation': f'Fallback prediction based on {len(similar_reviews)} similar reviews',
                'similar_reviews': similar_reviews
            }
    
    def _build_context(self, similar_reviews: List[Dict]) -> str:
        """Build context string from similar reviews"""
        context_parts = []
        for i, review in enumerate(similar_reviews, 1):
            context_parts.append(
                f"Similar Review {i}:\n"
                f"Decision: {review['decision']}\n"
                f"Rating: {review['rating']}\n"
                f"Review: {review['review_text'][:200]}...\n"
            )
        return "\n".join(context_parts)
    
    def _create_prediction_prompt(self, review_text: str, context: str, return_explanation: bool) -> str:
        """Create enhanced prompt for OpenAI prediction"""
        base_prompt = f"""
TASK: Predict the final decision (Accept/Reject) for a new paper review based on patterns from similar historical reviews.

HISTORICAL EVIDENCE - Similar Reviews from Database:
{context}

TARGET REVIEW TO ANALYZE:
Review Text: {review_text}

ANALYSIS INSTRUCTIONS:
1. Examine the language patterns, sentiment, and specific criticisms/praise in the target review
2. Compare these patterns with the historical reviews and their known outcomes
3. Look for key indicators such as:
   - Technical soundness mentions
   - Novelty and significance comments
   - Writing clarity feedback
   - Experimental rigor assessment
   - Comparison with prior work
   - Overall reviewer sentiment and confidence

4. Consider the rating patterns and decision outcomes from similar reviews
"""

        if return_explanation:
            base_prompt += """
REQUIRED OUTPUT FORMAT:
PREDICTION: [Accept/Reject]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Provide a detailed analysis covering:
- Key patterns you identified in the similar reviews
- Specific language/sentiment indicators that influenced your decision
- How the target review compares to accepted vs rejected papers
- What aspects of the review suggest the predicted outcome
- Any notable strengths or weaknesses that were decisive factors]
"""
        else:
            base_prompt += """
REQUIRED OUTPUT FORMAT:
PREDICTION: [Accept/Reject]
CONFIDENCE: [0.0-1.0]
"""

        return base_prompt
    
    def _parse_openai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse OpenAI response into structured format"""
        result = {
            'prediction': 'Reject',
            'confidence': 0.5,
            'explanation': 'Could not parse response'
        }

        # Split response into lines
        lines = response_text.strip().split('\n')

        # Find the explanation section
        explanation_started = False
        explanation_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Parse prediction
            if line_stripped.startswith('PREDICTION:'):
                pred = line_stripped.split(':', 1)[1].strip()
                if 'Accept' in pred:
                    result['prediction'] = 'Accept'
                else:
                    result['prediction'] = 'Reject'

            # Parse confidence
            elif line_stripped.startswith('CONFIDENCE:'):
                try:
                    conf = float(line_stripped.split(':', 1)[1].strip())
                    result['confidence'] = max(0.0, min(1.0, conf))
                except:
                    pass

            # Parse explanation (multi-line)
            elif line_stripped.startswith('EXPLANATION:'):
                explanation_started = True
                # Get the text after "EXPLANATION:" on the same line
                explanation_text = line_stripped.split(':', 1)[1].strip()
                if explanation_text:
                    explanation_lines.append(explanation_text)

            # Continue collecting explanation lines
            elif explanation_started and line_stripped:
                explanation_lines.append(line_stripped)

        # Join explanation lines
        if explanation_lines:
            result['explanation'] = '\n'.join(explanation_lines)

        return result
    
    def predict_batch(self, review_texts: List[str]) -> List[str]:
        """Predict decisions for a batch of reviews (for evaluation)"""
        predictions = []
        for i, review_text in enumerate(review_texts):
            if i % 10 == 0:
                print(f"Processing review {i+1}/{len(review_texts)}")
            
            result = self.predict_decision(review_text, return_explanation=False)
            predictions.append(result['prediction'])
        
        return predictions
    
    def evaluate(self, test_reviews: List[str], true_decisions: List[str]) -> Dict[str, Any]:
        """Evaluate the RAG system on test data"""
        print("Starting RAG evaluation...")
        
        # Get predictions
        predictions = self.predict_batch(test_reviews)
        
        # Convert to binary format for evaluation
        pred_binary = [1 if p == 'Accept' else 0 for p in predictions]
        true_binary = [1 if t == 'Accept' else 0 for t in true_decisions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_binary, pred_binary)
        report = classification_report(true_binary, pred_binary, 
                                     target_names=['Reject', 'Accept'], 
                                     output_dict=True)
        cm = confusion_matrix(true_binary, pred_binary)
        
        results = {
            'accuracy': accuracy,
            'precision': report['Accept']['precision'],
            'recall': report['Accept']['recall'],
            'f1_score': report['Accept']['f1-score'],
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'num_samples': len(test_reviews)
        }
        
        print(f"RAG Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        
        return results

def test_rag_system():
    """Test the RAG system with a sample review"""
    rag = DecisionPredictionRAG()
    
    # Test with a sample review
    sample_review = """
    This paper presents an interesting approach to neural machine translation. 
    The experimental results show improvements over baseline methods. 
    However, the writing could be clearer and some experimental details are missing.
    The contribution is incremental but solid.
    """
    
    print("Testing RAG system with sample review...")
    result = rag.predict_decision(sample_review)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Explanation: {result['explanation']}")
    print(f"Based on {len(result['similar_reviews'])} similar reviews")

if __name__ == "__main__":
    test_rag_system()
