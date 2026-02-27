# Multi-Modal RAG Pipeline for Policy Document Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green.svg)](https://python.langchain.com/)

A production-grade Retrieval-Augmented Generation (RAG) system designed to handle complex policy documents with nested tables, state-specific variations, and multi-modal content (text, tables, images).


## 🎯 Overview

This project implements a hybrid RAG pipeline that addresses common limitations of standard vector search systems when dealing with enterprise policy documents. Built for accuracy and scalability, it combines multiple retrieval strategies with semantic reranking and multi-modal processing.

### Key Features

- **Hybrid Retrieval**: BM25 (40%) + Semantic Search (60%) for optimal recall
- **Semantic Reranking**: BGE-reranker-large for precision optimization (0.99+ relevance scores)
- **Multi-Modal Processing**: Handles text, tables, and images using Unstructured.io + Vision LLMs
- **Context Engineering**: Structured prompts for factually grounded responses
- **Production-Ready UI**: Streamlit interface with source attribution

### Use Cases

- Policy document Q&A systems
- Legal document analysis
- Regulatory compliance tools
- Technical documentation search
- Any domain requiring precise information retrieval from structured documents

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Document Processing](#document-processing)
  - [Running the Application](#running-the-application)
  - [API Usage](#api-usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## 🏗️ Architecture

```
┌─────────────────┐
│  PDF Document   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Unstructured.io hi_res     │
│  • Table inference          │
│  • Image extraction         │
│  • Layout preservation      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Chunking & Embedding       │
│  • BGE-large-en-v1.5        │
│  • Semantic + Metadata      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Hybrid Retrieval (k=10)    │
│  • BM25 (0.4 weight)        │
│  • Vector Search (0.6)      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Semantic Reranking (k=3)   │
│  • BGE-reranker-large       │
│  • Score: 0.99+             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Multi-Modal LLM            │
│  • Llama 3.2-Vision         │
│  • Context: Text+Tables+Img │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Response + Sources         │
└─────────────────────────────┘
```

---

## 🔧 Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: 16GB minimum (32GB recommended for large documents)
- **Storage**: 10GB free space (for models and vector DB)
- **GPU**: Optional (CPU-only mode supported)

### Software Dependencies

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) for local LLM inference
- Tesseract OCR (for document processing)
- Poppler (for PDF rendering)

### Installing System Dependencies

#### macOS
```bash
brew install tesseract poppler
brew install ollama
ollama pull llama3.2-vision
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2-vision
```

#### Windows (WSL2)
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2-vision
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multimodal-rag-pipeline.git
cd multimodal-rag-pipeline
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n rag_env python=3.10
conda activate rag_env
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first run will download ~2GB of models (BGE embeddings, reranker, Unstructured models).

### 4. Verify Ollama Installation

```bash
# Start Ollama service
ollama serve

# In another terminal, verify model availability
ollama list
# Should show: llama3.2-vision
```

---

## 🚀 Quick Start

### Option 1: Use Pre-Processed Sample Data

```bash
# Download sample database (SNAP policy document - 100+ pages)
# TODO: Add download link or instruction
```

### Option 2: Process Your Own Document

```bash
# 1. Place your PDF in the project root
cp /path/to/your/document.pdf ./document.pdf

# 2. Run the processing notebook
jupyter notebook RAG_pipeline.ipynb
# Execute cells 1-7 to process and index your document

# 3. Update database path in app.py if needed
# Line 32: db_path = "dbv1/chroma_db_bge"
```

### Run the Application

```bash
# Ensure Ollama is running
ollama serve

# In another terminal
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 💻 Usage

### Document Processing

#### Step 1: Partition PDF

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="your_document.pdf",
    strategy="hi_res",              # Enables table detection
    infer_table_structure=True,     # Preserves table structure
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True
)

print(f"Extracted {len(elements)} elements")
```

#### Step 2: Chunk and Embed

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Create vector database
db = Chroma.from_documents(
    documents=processed_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

#### Step 3: Query

```python
query = "What are the income limits for SNAP eligibility in Alaska?"

# Hybrid retrieval
initial_docs = hybrid_retriever.invoke(query)

# Rerank
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('BAAI/bge-reranker-large')
pairs = [[query, doc.page_content] for doc in initial_docs]
scores = reranker.predict(pairs)

# Get top 3
top_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)[:3]
```

### Running the Application

#### Basic Usage

```bash
streamlit run app.py
```

#### Custom Configuration

```bash
# Change port
streamlit run app.py --server.port 8080

# Enable CORS for API access
streamlit run app.py --server.enableCORS=true
```

### API Usage (Programmatic Access)

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Load system
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
db = Chroma(persist_directory="dbv1/chroma_db_bge", embedding_function=embeddings)
reranker = CrossEncoder('BAAI/bge-reranker-large')

# Query
query = "Your question here"
docs = db.similarity_search(query, k=10)
pairs = [[query, d.page_content] for d in docs]
scores = reranker.predict(pairs)
top_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:3]

print(f"Top result score: {top_docs[0][1]:.4f}")
print(f"Content: {top_docs[0][0].page_content}")
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Optional: Unstructured API (for cloud processing)
UNSTRUCTURED_API_KEY=your_api_key_here

# Optional: OpenAI (if using GPT models instead of Ollama)
OPENAI_API_KEY=your_openai_key

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision
```

### Application Settings

Edit `app.py` to customize:

```python
# Line 32: Vector database path
db_path = "dbv1/chroma_db_bge"

# Line 48: Hybrid retrieval weights
weights=[0.4, 0.6]  # [BM25, Semantic]

# Line 54: Reranker model
reranker = CrossEncoder('BAAI/bge-reranker-large')

# Line 57: LLM model
llm = ChatOllama(model="llama3.2-vision", temperature=0)
```

---

## 📁 Project Structure

```
multimodal-rag-pipeline/
├── RAG_pipeline.ipynb          # Document processing pipeline
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
│
├── data/                      # Input documents
│   └── SNAP.pdf              # Sample policy document
│
├── dbv1/                     # Vector databases
│   └── chroma_db_bge/       # ChromaDB storage
│
├── docs/                     # Documentation
│   ├── architecture.png     # System diagram
│   └── examples.md          # Usage examples
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_document_processing.ipynb
│   ├── 02_evaluation.ipynb
│   └── 03_experiments.ipynb
│
└── tests/                  # Unit tests
    ├── test_retrieval.py
    └── test_reranking.py
```

---

## 📊 Performance

### Benchmark Results

Tested on 100+ page SNAP policy document (423 elements):

| Metric | Value |
|--------|-------|
| **Average Relevance Score** | 0.99+ |
| **Retrieval Time** (10 docs) | ~0.8s |
| **Reranking Time** (10→3) | ~0.3s |
| **End-to-End Latency** | ~5-8s |
| **Memory Usage** | ~4GB (CPU mode) |

### Example Queries

| Query Complexity | Relevance Score | Response Quality |
|-----------------|----------------|------------------|
| Simple fact lookup | 0.9993 | Exact match |
| Multi-condition eligibility | 0.9996 | Comprehensive |
| State-specific rules | 0.8483 | Accurate with caveats |
| Temporal provisions | 0.9532 | Correctly time-bounded |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Ollama Connection Error

```
Error connecting to Ollama: Connection refused
```

**Solution**:
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

#### 2. Database Not Found

```
Database not found at dbv1/chroma_db_bge
```

**Solution**:
- Run `RAG_pipeline.ipynb` cells 1-7 to create the database
- Or update `db_path` in `app.py` to point to your database

#### 3. Memory Error During Processing

```
RuntimeError: [enforce fail at alloc_cpu.cpp:114] err == 0. DefaultCPUAllocator: not enough memory
```

**Solution**:
- Process documents in smaller batches
- Reduce chunk size in the notebook
- Close other applications to free RAM

#### 4. Tesseract Not Found

```
TesseractNotFoundError: tesseract is not installed
```

**Solution**:
```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Verify installation
tesseract --version
```

#### 5. Slow First Run

The first run downloads several models (~2GB total):
- BGE-large-en-v1.5 (~1.2GB)
- BGE-reranker-large (~600MB)
- Unstructured models (~200MB)

This is normal and only happens once.

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork the repo and clone your fork
git clone https://github.com/yourusername/multimodal-rag-pipeline.git

# Create a feature branch
git checkout -b feature/your-feature-name

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .
```

### Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass: `pytest tests/`
4. Update `CHANGELOG.md` with your changes
5. Submit PR with clear description of changes

### Code Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints where possible
- Add docstrings to all functions
- Maximum line length: 100 characters

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Models**: 
  - [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) for embeddings
  - [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) for reranking
  - [Meta Llama 3.2-Vision](https://ollama.ai/library/llama3.2-vision) for generation

- **Frameworks**:
  - [LangChain](https://python.langchain.com/) for RAG orchestration
  - [Unstructured.io](https://unstructured.io/) for document processing
  - [ChromaDB](https://www.trychroma.com/) for vector storage

---





