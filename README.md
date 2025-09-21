Financial RAG System with Agent Capabilities
This is a Retrieval-Augmented Generation (RAG) system designed to analyze and answer complex questions about financial 10-K documents. It uses an autonomous agent to decompose multi-step queries and a vector database to retrieve relevant information from a corpus of financial reports.

Features
Document Processing: Automatically extracts and chunks text from 10-K PDFs in the data/ directory.

Vector Store: Utilizes a FAISS-based vector store for efficient similarity search of financial document chunks.

Semantic Search: Employs the Sentence-Transformers model (all-MiniLM-L6-v2) to create high-quality embeddings for accurate retrieval.

Query Decomposition: A custom QueryDecomposer agent identifies complex, multi-step queries (e.g., cross-company comparisons or year-over-year growth) and breaks them down into simpler, actionable sub-queries.

Autonomous Agent: The system leverages a LangChain ReAct agent to reason about the user's request and effectively use the FinancialRAGTool to retrieve information.

JSON Output: All results are returned in a structured JSON format, including the final answer, reasoning, sub-queries, and cited sources from the documents.

Setup
Prerequisites
Python 3.8 or higher

A Gemini API Key (from Google AI Studio)

Installation
Clone the repository:

git clone [https://github.com/Poojav21/FinQAgent-Finance-Query-Agent-.git]
cd FinQAgent-Finance-Query-Agent-

Create and activate a virtual environment:

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install the required packages. The following packages are required for the project:

google-generativeai
faiss-cpu
sentence-transformers
pypdf2
numpy
langchain
langchain-core
langchain-community
langchain-google-genai

You can create a requirements.txt file with the above packages and run:

pip install -r requirements.txt

Add your Gemini API key to the FinancialRAGSystem class in rag.py. You must replace "YOUR_ACTUAL_GEMINI_API_KEY" with your key.

Data
Place your financial 10-K PDFs in the data/ directory. The script expects the files to be organized by company and year, like this:

data/
└── google/
    ├── 2022.pdf
    └── 2023.pdf
└── nvidia/
    ├── 2022.pdf
    └── 2023.pdf

Usage
Run the main script from your terminal:

python rag.py

The system will first set up by either loading an existing vector store or building a new one from your PDF documents.

You will then be prompted to choose between running predefined test queries or entering interactive mode.

Interactive Mode
Simply type your financial question and press Enter. The system will process the query, retrieve relevant information, and provide a comprehensive answer in JSON format.

Your question: How much did Microsoft's cloud revenue grow from 2022 to 2023?

The output will be a detailed JSON object containing the final answer, the breakdown of sub-queries, and the original document excerpts used as sources.
