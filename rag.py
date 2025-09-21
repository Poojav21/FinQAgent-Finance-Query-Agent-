"""
Financial RAG System with Agent Capabilities
Focused implementation for 10-K document analysis with JSON output format
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import logging
import google.generativeai as genai
import faiss
import pickle
import PyPDF2
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument
from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain import hub
from langchain.tools import BaseTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLM(LLM):
    """Simple Gemini LLM wrapper for LangChain"""
    model: Any = None 

    def __init__(self, api_key: str, **kwargs: Any):
        super().__init__(**kwargs) 
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: List[str] = None, **kwargs: Any) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {e}")
            return "An error occurred while generating the response."


class PDFProcessor:
    """Process PDF files and extract text"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "],
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"[PAGE {page_num + 1}] {page_text}\n"
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return ""
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def process_all_pdfs(self) -> List[LangChainDocument]:
        """Process all PDFs in data directory"""
        documents = []
        
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return documents

        for company_dir in self.data_dir.iterdir():
            if not company_dir.is_dir():
                continue
            
            company = company_dir.name
            logger.info(f"Processing {company}...")
            
            for pdf_file in company_dir.glob("*.pdf"):
                year = pdf_file.stem
                
                text = self.extract_text_from_pdf(str(pdf_file))
                if text:
                    chunks = self.text_splitter.split_text(text)
                    
                    for i, chunk in enumerate(chunks):
                        # Extract page number from chunk if available
                        page_match = re.search(r'\[PAGE (\d+)\]', chunk)
                        page_num = int(page_match.group(1)) if page_match else 1
                        
                        doc = LangChainDocument(
                            page_content=chunk,
                            metadata={
                                "company": company,
                                "year": year,
                                "chunk_id": f"{company}_{year}_chunk_{i}",
                                "page": page_num,
                                "source": str(pdf_file)
                            }
                        )
                        documents.append(doc)
        
        logger.info(f"Created {len(documents)} document chunks")
        return documents

class VectorStore:
    """Simple FAISS vector store"""
    
    def __init__(self):
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
    
    def add_documents(self, documents: List[LangChainDocument]):
        """Add documents to vector store"""
        if not documents:
            logger.warning("No documents to add to vector store.")
            return

        texts = [doc.page_content for doc in documents]
        
        # Create embeddings
        embeddings = self.embeddings.encode(texts)
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        self.documents = documents
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[LangChainDocument, float]]:
        """Search for similar documents"""
        if self.index is None:
            logger.error("Vector store not initialized. Run setup() first.")
            return []
            
        query_embedding = self.embeddings.encode([query]).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents): # Safety check
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """Save vector store"""
        os.makedirs(path, exist_ok=True)
        try:
            faiss.write_index(self.index, f"{path}/index.faiss")
            with open(f"{path}/documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load(self, path: str) -> bool:
        """Load vector store"""
        try:
            self.index = faiss.read_index(f"{path}/index.faiss")
            with open(f"{path}/documents.pkl", 'rb') as f:
                self.documents = pickle.load(f)
            return True
        except FileNotFoundError:
            logger.warning("Vector store files not found. A new one will be built.")
            return False
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False

class QueryDecomposer:
    """Agent for decomposing complex queries into sub-queries"""
    
    def __init__(self, llm: GeminiLLM):
        self.llm = llm
    
    def detect_query_type(self, query: str) -> str:
        """Detect the type of query to determine decomposition strategy"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["compare", "across", "all three", "which company"]):
            return "cross_company"
        elif any(phrase in query_lower for phrase in ["from 2022 to 2023", "grow from", "change from"]):
            return "yoy_comparison"
        elif any(word in query_lower for word in ["percentage", "came from", "segment"]):
            return "segment_analysis"
        else:
            return "simple"
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose query into sub-queries based on type"""
        query_type = self.detect_query_type(query)
        
        if query_type == "simple":
            return [query]
        
        elif query_type == "cross_company":
            companies = ["Microsoft", "Google", "NVIDIA"]
            if "operating margin" in query.lower():
                return [f"{company} operating margin 2023" for company in companies]
            elif "gross margin" in query.lower():
                return [f"{company} gross margin 2023" for company in companies]
            elif "revenue" in query.lower():
                return [f"{company} total revenue 2023" for company in companies]
            elif "r&d" in query.lower():
                return [f"{company} R&D spending 2023" for company in companies]
            else:
                return [f"{company} " + query.split("company")[-1].strip() for company in companies]
        
        elif query_type == "yoy_comparison":
            companies = ["Microsoft", "Google", "NVIDIA"]
            company = next((c for c in companies if c.lower() in query.lower()), "Microsoft")
            
            if "data center" in query.lower():
                return [f"{company} data center revenue 2022", f"{company} data center revenue 2023"]
            elif "cloud" in query.lower():
                return [f"{company} cloud revenue 2022", f"{company} cloud revenue 2023"]
            elif "revenue" in query.lower():
                return [f"{company} total revenue 2022", f"{company} total revenue 2023"]
            
        return [query]

class FinancialRAGTool(BaseTool):
    """LangChain tool for financial document search"""
    
    # Use BaseTool for a simpler Pydantic model setup.
    name: str = "financial_search"
    description: str = "Search financial 10-K documents for specific company and year data"
    vector_store: Any

    def _run(self, query: str) -> str:
        """Search for financial information"""
        if not self.vector_store:
            return "Vector store not initialized."

        results = self.vector_store.search(query, k=3)
        
        if not results:
            return "No relevant information found"
        
        context = []
        for doc, score in results:
            context.append(f"Company: {doc.metadata['company']}, Year: {doc.metadata['year']}\n{doc.page_content[:300]}...")
        
        return "\n\n".join(context)

class FinancialRAGSystem:
    """Main RAG system with agent capabilities"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.pdf_processor = PDFProcessor(data_dir)
        self.vector_store = VectorStore()
        
        # Initialize Gemini LLM
        # IMPORTANT: Replace with your actual Gemini API Key
        # This key is just an example and will not work.
        api_key = "sdhakskjsdhfkjsdhflasafkjagfkjdsgfa" 
        self.llm = GeminiLLM(api_key=api_key)
        
        self.decomposer = QueryDecomposer(self.llm)
        self.agent = None
    
    def setup(self, force_rebuild: bool = False):
        """Setup the RAG system"""
        vector_store_path = "vector_store"
        
        if not force_rebuild and self.vector_store.load(vector_store_path):
            logger.info("Loaded existing vector store")
        else:
            logger.info("Building new vector store...")
            documents = self.pdf_processor.process_all_pdfs()
            self.vector_store.add_documents(documents)
            self.vector_store.save(vector_store_path)
        
        # Setup agent with tools
        # Instantiate the tool using Pydantic's keyword argument initialization
        financial_tool = FinancialRAGTool(vector_store=self.vector_store)
        tools = [financial_tool]
        
        try:
            prompt = hub.pull("hwchase17/react")
            agent = create_react_agent(self.llm, tools, prompt)
            self.agent = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )
        except Exception as e:
            logger.warning(f"Agent setup failed: {e}. Using fallback method.")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query and return JSON response"""
        # Decompose query
        sub_queries = self.decomposer.decompose_query(question)
        
        # Execute sub-queries
        sub_results = []
        sources = []
        
        for sub_query in sub_queries:
            # Search documents
            results = self.vector_store.search(sub_query, k=3)
            
            if results:
                # Get best result
                doc, score = results[0]
                
                # Generate answer for this sub-query
                context = f"Company: {doc.metadata['company']}, Year: {doc.metadata['year']}\n{doc.page_content}"
                
                prompt = f"""
                Based on this financial document excerpt, answer the specific question.
                
                Question: {sub_query}
                
                Document excerpt:
                {context}
                
                Provide a specific, numerical answer when possible. If the information isn't available, say so.
                """
                
                answer = self.llm._call(prompt)
                sub_results.append(answer)
                
                # Add to sources
                sources.append({
                    "company": doc.metadata["company"],
                    "year": doc.metadata["year"],
                    "excerpt": doc.page_content[:150] + "...",
                    "page": doc.metadata.get("page", 1),
                    "score": score
                })
        
        # Synthesize final answer
        if len(sub_queries) > 1:
            synthesis_prompt = f"""
            Original question: {question}
            
            Sub-questions and answers:
            {chr(10).join([f"Q: {sq} A: {sr}" for sq, sr in zip(sub_queries, sub_results)])}
            
            Provide a comprehensive answer to the original question by synthesizing these results.
            Include specific numbers and comparisons where possible.
            """
            
            final_answer = self.llm._call(synthesis_prompt)
            reasoning = f"Decomposed query into {len(sub_queries)} sub-questions and synthesized results"
        else:
            final_answer = sub_results[0] if sub_results else "No relevant information found"
            reasoning = "Direct query execution"
        
        # Return JSON format as specified
        return {
            "query": question,
            "answer": final_answer,
            "reasoning": reasoning,
            "sub_queries": sub_queries,
            "sources": sources
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.vector_store.documents:
            return {
                "total_documents": 0,
                "companies": [],
                "years": []
            }
        companies = set(doc.metadata["company"] for doc in self.vector_store.documents)
        years = set(doc.metadata["year"] for doc in self.vector_store.documents)
        
        return {
            "total_documents": len(self.vector_store.documents),
            "companies": sorted(list(companies)),
            "years": sorted(list(years))
        }

def run_test_queries():
    """Run the specified test queries"""
    test_queries = [
        # Simple queries
        "What was NVIDIA's total revenue in fiscal year 2024?",
        "What percentage of Google's 2023 revenue came from advertising?",
        
        # Comparative queries (require agent decomposition)
        "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
        "Which of the three companies had the highest gross margin in 2023?",
        
        # Complex multi-step queries
        "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
    ]
    
    rag = FinancialRAGSystem()
    rag.setup()
    
    print("Financial RAG System Test Results")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. {query}")
        print("-" * 30)
        
        result = rag.query(query)
        
        # Pretty print JSON
        print(json.dumps(result, indent=2))

def main():
    """Main interactive function"""
    print("Financial RAG System with Agent Capabilities")
    print("=" * 50)
    
    rag = FinancialRAGSystem()
    
    print("Setting up system...")
    rag.setup()
    
    stats = rag.get_stats()
    print(f"System ready! Loaded {stats['total_documents']} documents")
    print(f"Companies: {', '.join(stats['companies'])}")
    print(f"Years: {', '.join(stats['years'])}")
    print("\n" + "=" * 50)
    
    # Option to run test queries
    choice = input("Run test queries? (y/n): ").lower()
    if choice == 'y':
        run_test_queries()
        return
    
    # Interactive mode
    print("Interactive Mode (type 'quit' to exit):")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        
        print("Processing...")
        result = rag.query(question)
        
        # Pretty print result
        print("\nResult:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()