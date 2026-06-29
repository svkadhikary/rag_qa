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

## Tech Stack

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

# Set up API keys in SECRETS.txt (set in env for security):
GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key
```
## Usage
```python app.py```

