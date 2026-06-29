# RAG PDF Demo

A Retrieval-Augmented Generation (RAG) application that allows you to chat with PDF documents or web content using vector search and LLM-powered responses.

## Key Features

- **PDF Processing** – Upload PDFs, split into chunks, and generate vector embeddings
- **URL Support** – Fetch and process text content from web URLs
- **Intelligent Query Handling** – Query rewriting for better retrieval accuracy
- **Vector Search** – FAISS-based similarity search with reranking
- **RAG Pipeline** – Retrieve relevant chunks, rerank with Cohere, generate responses via Groq
- **Interactive UI** – Gradio-based chat interface with retrieval transparency
- **Evaluation Metrics** – Displays faithfulness, recall, and reasoning scores from latest evaluations
- **Logging** - Logs every stages of the workflow

## Tech Stack
- **Language**: Python==3.11+
- **Framework**: Gradio (UI)
- **LLM**: Groq (response generation & query rewriting)
- **Reranker**: Cohere
- **Vector Store**: FAISS (via custom VectorStore module)
- **Embeddings**: Custom transformer-based embeddings (TransformerUtils)
- **Data Processing**: PyPDF, LangChain text splitters

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Download the embedding model locally
# Place model in ./local_minilm directory
```python TransformerUtils\download_model.py``` ## downloads the allminiLM model and saves in local_minilm

# Set up API keys in SECRETS.txt (set in env for security):
GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key
```
## Usage
```python app.py```

## Steps:

- Upload a PDF or enter URLs to process
- Wait for document processing (chunking → embedding → indexing)
- Ask questions in the chat interface
- View retrieved chunks in the accordion panel

## How It Works
 - **Document Processing** – PDF/URL content is loaded and split into chunks (default: 400 chars, 50 overlap)
 - **Embedding** – Chunks are transformed into vector representations using MiniLM model (mean pooling + normalization)
 - **Indexing** – Vectors are stored in a FAISS index using L2 distance for fast similarity search
 - **Query Rewriting** – User question is enriched by Groq for better retrieval context
 - **Retrieval** – Top-k relevant chunks are fetched from FAISS
 - **Reranking** – Cohere reranks chunks for improved relevance
 - **Generation** – Groq LLM (Llama 3.3 70B) generates response using reranked context

## LLM Integration (Groq)
 #### The app uses Groq's Llama 3.3 70B model for:
  - **Query Rewriting** – Optimizes user queries for RAG retrieval
  - **Response Generation** – Answers questions using retrieved context
  - **Evaluation** – Generates test Q&A pairs for RAG evaluation

## Configuration:
 - **Default model**: llama-3.3-70b-versatile
 - **Temperature**: 0.2 (generation), 0.5 (query rewrite)
 - **Max tokens**: 1024

## Reranking (Cohere)
 - **Model**: rerank-v4.0-pro
 - **Purpose**: Re-orders retrieved chunks by relevance to the query
 - **Output**: Returns top-N most relevant documents (default: 5)

## Vector Store (FAISS)
 - **Index Type**: IndexFlatL2 (exact L2 distance search)
 - **Normalization**: L2 normalization applied to vectors before indexing
 - **Search**: Returns top-k nearest neighbors with distances
 - **Scalability**: Works for moderate-sized document collections

## Embedding Model
 ##### The app uses a local MiniLM model (./local_minilm) for generating embeddings:
  - **Model**: Sentence Transformers (allMiniLM)
  - **Method**: Mean pooling + L2 normalization
  - **Output**: 384-dimension vectors (or model-specific dimension)
  - **Local Only**: Runs offline without API calls

### Configuration
 ##### Key adjustable parameters in app.py:
 - ```chunk_size=400``` – PDF chunk size
 - ```overlap=50``` – Chunk overlap
 - ```top_k=10``` – Retrieved chunks before reranking
 - ```temperature=0.0-1.0``` – LLM temperature settings

### For Evaluation:
 - ```python evaluate.py``` runs the evaluation script on the test_dataset.json
 - It uses the same workflow as original application.
 - Saves the evaluation results in eval_results/
 - Uncomment the marked code block to create test_dateset from the doccument of your choice