# ASAP Review Decision Prediction: Comprehensive BERT vs RAG Analysis

This project implements and compares **three distinct approaches** for predicting paper acceptance/rejection decisions based on peer reviews from the ASAP-Review dataset:

1. **Untrained BERT Baseline**: Baseline performance using pre-trained BERT without fine-tuning
2. **Fine-tuned BERT**: Supervised learning with BERT fine-tuned on academic review data
3. **RAG System**: Retrieval-Augmented Generation using Qdrant vector database and OpenAI GPT-3.5-turbo

##  Complete Experimental Results

### Final Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Jaccard | Approach |
|-------|----------|-----------|--------|----------|---------|----------|
| **Untrained BERT** | 39.79% | 66.67% | 0.29% | 0.58% | 39.79% | Baseline |
| **Fine-tuned BERT** | **82.53%** | **84.65%** | **86.75%** | **85.69%** | **82.53%** | Supervised Learning |
| **RAG System** | 49.50% | 76.47% | 21.85% | 33.99% | 49.50% | Retrieval + Generation |

### Key Findings
- **🏆 Winner**: Fine-tuned BERT significantly outperforms both baseline and RAG approaches
- **📈 Performance Gap**: 85.11% F1-score difference between Fine-tuned BERT and Untrained BERT
- **🎯 Untrained BERT**: Extremely low recall (0.29%) shows inability to detect positive cases without training
- **⚡ Efficiency**: BERT models are 6x faster than RAG system for inference



### RAG System Implementation

1. **Start Qdrant Vector Database**:
   ```bash
   # Open Docker Desktop first
   docker pull qdrant/qdrant
   docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
   ```

   Qdrant dashboard: http://localhost:6333/dashboard

2. **Ingest Data to Vector Database**:
   ```bash
   # Full dataset ingestion
   python qdrant_pipeline/rag_ingest.py
   ```

3. **Run RAG Predictions**:
   ```bash
   python qdrant_pipeline/rag_predictor.py
   ```

4. **Run RAG Evaluation**:
   ```bash
   python qdrant_pipeline/rag_evaluation.py
   ```

## 📈 Detailed Results Analysis

### BERT Fine-tuning Performance

The BERT model showed excellent performance on the ASAP review classification task:

**Key Observations**:
- Consistent improvement across all epochs
- Strong precision (84.65%) indicates low false positive rate
- Excellent recall (86.75%) shows outstanding positive case detection
- F1-score of 85.69% demonstrates superior balanced performance
- Dramatic improvement over untrained baseline (85.11% F1-score gain)

### RAG System Architecture

**Components**:
1. **Vector Database**: Qdrant with 34,046+ review documents
2. **Embeddings**: BAAI/bge-large-en (1024-dimensional)
3. **Retrieval**: Semantic similarity search (top-5 reviews)
4. **Generation**: OpenAI GPT-3.5-turbo with enhanced prompts
5. **Fallback**: Majority voting when API fails


### Complete Three-Way Comparative Analysis

| Model | Accuracy | Precision | Recall | F1-Score | Jaccard | Inference Time | Approach |
|-------|----------|-----------|--------|----------|---------|----------------|----------|
| **Untrained BERT** | 39.79% | 66.67% | 0.29% | 0.58% | 39.79% | ~2 min | Baseline |
| **Fine-tuned BERT** | **82.53%** | **84.65%** | **86.75%** | **85.69%** | **82.53%** | ~3 min | Supervised Learning |
| **RAG System** | 49.50% | 76.47% | 21.85% | 33.99% | 49.50% | ~20 min | Retrieval + Generation |

### Performance Analysis:
- **🏆 Clear Winner**: Fine-tuned BERT dominates all metrics
- **📉 Untrained BERT**: Catastrophically low recall (0.29%) - essentially fails to detect positive cases
- **🎯 RAG Trade-offs**: High precision (76.47%) but low recall (21.85%) - conservative predictions
- **⚡ Efficiency**: BERT models are significantly faster than RAG
- **📊 Practical Impact**: 85.11% F1-score gap between Fine-tuned and Untrained BERT


##  Key Findings and Insights

### Performance Hierarchy:
1. **Fine-tuned BERT** (85.69% F1) - Clear winner with excellent balanced performance
2. **RAG System** (33.99% F1) - Moderate performance with conservative bias
3. **Untrained BERT** (0.58% F1) - Catastrophic failure, essentially non-functional

###  Critical Insights:

#### BERT Strengths:
- **Exceptional Performance**: 82.53% accuracy on challenging academic review task
- **Excellent Metrics**: Outstanding precision-recall balance (84.65% / 86.75%)
- **High Recall**: 86.75% recall shows excellent positive case detection
- **Computational Efficiency**: 6x faster inference than RAG system
- **Reproducibility**: Deterministic results, no external API dependencies
- **Cost Effectiveness**: One-time training cost vs. ongoing API costs

#### RAG System Characteristics:
- **Conservative Bias**: High precision (76.47%) but very low recall (21.85%)
- **Risk-Averse Behavior**: Tends to predict "Reject" when uncertain
- **Interpretability Advantage**: Provides reasoning and similar review examples
- **Context Awareness**: Leverages historical review patterns effectively
- **Flexibility**: No retraining needed for new data

#### Unexpected Findings:
- **Untrained BERT Catastrophe**: Performed far worse than expected (0.58% F1 vs. expected ~50%)
- **Extreme Class Imbalance Impact**: Untrained BERT shows severe bias toward majority class
- **RAG Moderate Performance**: Despite theoretical advantages, RAG achieved moderate results
- **Retrieval Limitations**: Similar reviews may not always provide relevant decision context
- **Generation Inconsistency**: LLM outputs varied despite low temperature settings

### 🔄 Trade-offs Analysis:
- **Performance vs. Interpretability**: BERT wins on metrics, RAG wins on explainability
- **Speed vs. Context**: BERT is faster, RAG provides richer context
- **Cost vs. Flexibility**: BERT has upfront training cost, RAG has ongoing API costs
- **Determinism vs. Adaptability**: BERT is deterministic, RAG adapts to new patterns
'''
