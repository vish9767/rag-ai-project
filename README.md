# 🧠 RAG AI - Retrieval-Augmented Generation from PDFs

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using LangChain, OpenAI, and Pinecone to answer questions from the content of a PDF document.

It allows you to load a PDF file, split it into chunks, embed the text using OpenAI's embedding models, store and retrieve it from a Pinecone vector database, and then generate accurate answers using a large language model (LLM).

---

## 🚀 Features

- 📄 Load and parse PDFs using `langchain_community.document_loaders.PyPDFLoader`
- ✂️ Split documents into chunks using `RecursiveCharacterTextSplitter`
- 🔍 Embed chunks with `OpenAIEmbeddings`
- 🧠 Store and search embeddings using `PineconeVectorStore`
- 🤖 Answer questions with `ChatOpenAI` using context retrieved from PDF
- 🪄 Uses RAG (Retrieval-Augmented Generation) technique
- ⚡ Asynchronous and optimized for speed with Pinecone multithreaded indexing

---

## 📦 Tech Stack

| Tool         | Purpose                             |
|--------------|-------------------------------------|
| LangChain    | Framework for RAG pipelines         |
| OpenAI       | Embeddings + ChatGPT LLM            |
| Pinecone     | Vector database for semantic search |
| PyPDFLoader  | PDF document parsing                |
| Python       | Language used                       |

---

## 📁 Folder Structure (Optional Example)


rag-ai-project/
│
├── rag_app.py # Main script (your code)
├── requirements.txt # Python dependencies
├── README.md # Project overview
└── sample.pdf # Example PDF for testing



---



