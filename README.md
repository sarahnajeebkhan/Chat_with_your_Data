
# PDF-Based Chatbot using RAG with Qwen LLM and Chroma DB

This project implements a **Retrieval-Augmented Generation (RAG)**-based chatbot capable of answering questions from a set of uploaded PDFs. It leverages the **Qwen Large Language Model (LLM)** for natural language understanding and generation, and **Chroma DB** as a vector store for efficient document retrieval.

---

## Features

- **Upload PDFs**: Users can upload PDF documents as the knowledge base.
- **RAG-based Chatbot**: Combines retrieval of relevant document chunks with the generative capabilities of Qwen LLM.
- **Efficient Search**: Uses Chroma DB for embedding storage and fast similarity-based search.
- **Interactive Responses**: Provides contextually relevant answers to user queries.

---

## Workflow

1. **PDF Preprocessing**:
   - Extract text from PDFs.
   - Chunk text into manageable sections for vectorization.

2. **Embedding Creation**:
   - Generate embeddings for text chunks using a compatible embedding model.
   - Store embeddings in Chroma DB for efficient retrieval.

3. **Chatbot Interaction**:
   - User query is embedded and compared with stored embeddings in Chroma DB.
   - The most relevant chunks are retrieved and passed to the Qwen LLM for answer generation.

---

## Installation

### Prerequisites

- Python 3.8+
- `pip` package manager

### Dependencies

Install required Python libraries:
```bash
pip install langchain langchain-community chromadb pypdf
