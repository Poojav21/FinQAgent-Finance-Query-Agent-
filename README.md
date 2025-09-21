# 📊 Financial RAG System with Agent Capabilities

This repository implements a **Financial Retrieval-Augmented Generation (RAG) system** tailored for analyzing **10-K SEC filings** of major tech companies like **Microsoft, Google, and NVIDIA**.
It has **two main components**:

1. **Data Extraction Pipeline** → Downloads, processes, and chunks SEC 10-K filings into structured JSON format.
2. **RAG + Agent System** → Uses embeddings, FAISS vector search, and Gemini-powered agents to answer financial queries in natural language.

---

## 🚀 Features

✅ **Automated SEC Filings Downloader** (HTML/PDF)
✅ **Text Extraction & Chunking** (BeautifulSoup, PyPDF2)
✅ **Document Structuring** into JSON with metadata (company, year, section, etc.)
✅ **Embeddings & Vector Store** using SentenceTransformers + FAISS
✅ **Gemini LLM Wrapper** integrated with LangChain
✅ **Query Decomposition Agent** for multi-step reasoning (cross-company, YoY, etc.)
✅ **JSON-formatted Responses** with sources, reasoning, and extracted context

---

## 📂 Project Structure

```
├── data/                     # Downloaded SEC filings + processed docs
│   ├── GOOGL_2023_10K.html
│   ├── MSFT_2024_10K.pdf
│   └── processed_documents.json
│
├── vector_store/             # Saved FAISS index + documents
│   ├── index.faiss
│   └── documents.pkl
│
├── data_extraction.py         # SEC data extraction pipeline
├── rag.py              # RAG system with agent capabilities
├── requirements.txt
└── README.md
```

---

## 🏗️ Part 1: Data Extraction Pipeline

File: `sec_downloader.py`

### Workflow:

1. **Download SEC 10-K filings** (PDF or HTML fallback).
2. **Extract clean text** using BeautifulSoup (HTML) or PyPDF2 (PDF).
3. **Chunk text** into overlapping segments for embeddings.
4. **Save structured JSON** with company, year, and chunk metadata.

### Example Output (`processed_documents.json`):

```json
[
  {
    "content": "Alphabet Inc. is a collection of companies...",
    "company": "GOOGL",
    "year": "2023",
    "filing_type": "10-K",
    "page": null,
    "section": null,
    "chunk_id": "GOOGL_2023_0"
  }
]
```

Run:

```bash
python data_extraction.py
```

---

## 🧠 Part 2: RAG Agent System

File: `rag.py`

### Components:

* **PDFProcessor** → Converts 10-K PDFs into chunks for embedding.
* **VectorStore** → FAISS-based similarity search.
* **GeminiLLM** → Wrapper around Gemini API for natural language reasoning.
* **QueryDecomposer** → Splits complex questions into sub-queries.
* **FinancialRAGTool** → LangChain-compatible tool for financial retrieval.
* **FinancialRAGSystem** → Orchestrates RAG pipeline + agent execution.

### Example Query Flow:

**Question:**

```
"Compare the R&D spending as a percentage of revenue across all three companies in 2023"
```

**Decomposition:**

* "Microsoft R\&D spending as % of revenue 2023"
* "Google R\&D spending as % of revenue 2023"
* "NVIDIA R\&D spending as % of revenue 2023"

**Answer (JSON):**

```json
{
  "query": "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
  "answer": "In 2023, Microsoft spent 13% of revenue on R&D, Google 15%, and NVIDIA 20%. NVIDIA had the highest R&D intensity.",
  "reasoning": "Decomposed query into 3 sub-questions and synthesized results",
  "sub_queries": [
    "Microsoft R&D spending 2023",
    "Google R&D spending 2023",
    "NVIDIA R&D spending 2023"
  ],
  "sources": [
    {"company": "MSFT", "year": "2023", "excerpt": "R&D expenses were...", "page": 45, "score": 0.92},
    {"company": "GOOGL", "year": "2023", "excerpt": "Our R&D spending...", "page": 32, "score": 0.88},
    {"company": "NVDA", "year": "2023", "excerpt": "Research and development costs...", "page": 27, "score": 0.90}
  ]
}
```

---

## ⚡ Usage

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Download Filings & Process

```bash
python data_extraction.py
```

### 3. Build RAG System & Query

```bash
python rag.py
```

* Choose to run **test queries** (predefined)
* Or ask interactively:

```bash
Your question: What was NVIDIA's total revenue in fiscal year 2024?
```

---

## ✅ Example Test Queries

* "What was NVIDIA's total revenue in fiscal year 2024?"
* "What percentage of Google's 2023 revenue came from advertising?"
* "How much did Microsoft's cloud revenue grow from 2022 to 2023?"
* "Which of the three companies had the highest gross margin in 2023?"
* "Compare the R\&D spending as a percentage of revenue across all three companies in 2023"

---

## 🔮 Future Improvements

* Support **more SEC filing types** (10-Q, 8-K).
* Add **financial ratio calculators** as tools.
* Enhance **multi-hop reasoning** across documents.
* Integrate **streamlit/Gradio UI** for interactive dashboards.


