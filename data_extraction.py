# import os
# import json
# import re
# import requests
# from typing import List, Dict, Any, Optional
# from dataclasses import dataclass
# from pathlib import Path
# import logging
# from bs4 import BeautifulSoup
# import time

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class Document:
#     """Represents a document chunk with metadata"""
#     content: str
#     company: str
#     year: str
#     filing_type: str
#     page: Optional[int] = None
#     section: Optional[str] = None
#     chunk_id: Optional[str] = None

# class SECFilingDownloader:
#     """Downloads 10-K filings from SEC EDGAR"""
    
#     def __init__(self, data_dir: str = "data"):
#         self.data_dir = Path(data_dir)
#         self.data_dir.mkdir(exist_ok=True)
#         self.session = requests.Session()
#         self.session.headers.update({
#             'User-Agent': 'Mozilla/5.0 (compatible; Financial RAG System 1.0; academic-research@example.com)',
#             'Accept-Encoding': 'gzip, deflate',
#             'Accept': 'application/json, text/html',
#             'Connection': 'keep-alive'
#         })
        
#         # Company info with correct CIKs
#         self.companies = {
#             'GOOGL': {'cik': '1652044', 'name': 'Alphabet Inc.'},
#             'MSFT': {'cik': '789019', 'name': 'Microsoft Corporation'},
#             'NVDA': {'cik': '1045810', 'name': 'NVIDIA Corporation'}
#         }
    
#     def get_filing_urls(self, cik: str, form_type: str = "10-K") -> List[Dict]:
#         """Get filing URLs for a company"""
#         # Ensure CIK is 10 digits with leading zeros
#         formatted_cik = cik.zfill(10)
#         url = f"https://data.sec.gov/submissions/CIK{formatted_cik}.json"
        
#         logger.info(f"Fetching filings from: {url}")
        
#         try:
#             response = self.session.get(url, timeout=30)
#             response.raise_for_status()
#             data = response.json()
            
#             filings = []
#             recent = data.get('filings', {}).get('recent', {})
            
#             if not recent:
#                 logger.warning(f"No recent filings found for CIK {cik}")
#                 return []
            
#             forms = recent.get('form', [])
#             filing_dates = recent.get('filingDate', [])
#             accession_numbers = recent.get('accessionNumber', [])
#             primary_documents = recent.get('primaryDocument', [])
            
#             for i, form in enumerate(forms):
#                 if form == form_type and i < len(filing_dates):
#                     filing_date = filing_dates[i]
#                     year = int(filing_date[:4])
#                     if year in [2022, 2023, 2024]:
#                         if i < len(accession_numbers) and i < len(primary_documents):
#                             accession = accession_numbers[i].replace('-', '')
#                             primary_doc = primary_documents[i]
                            
#                             filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
                            
#                             filings.append({
#                                 'year': year,
#                                 'url': filing_url,
#                                 'accession': accession,
#                                 'filing_date': filing_date,
#                                 'primary_document': primary_doc
#                             })
            
#             logger.info(f"Found {len(filings)} {form_type} filings for years 2022-2024")
#             return sorted(filings, key=lambda x: x['year'])
        
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Network error fetching filings for CIK {cik}: {e}")
#             return []
#         except KeyError as e:
#             logger.error(f"Unexpected data structure for CIK {cik}: missing key {e}")
#             return []
#         except Exception as e:
#             logger.error(f"Error fetching filings for CIK {cik}: {e}")
#             return []
    
#     def download_filing(self, company: str, filing_info: Dict) -> Optional[str]:
#         """Download a single filing"""
#         file_path = self.data_dir / f"{company}_{filing_info['year']}_10K.html"
        
#         if file_path.exists():
#             logger.info(f"File already exists: {file_path}")
#             return str(file_path)
        
#         try:
#             logger.info(f"Downloading {company} {filing_info['year']} 10-K from {filing_info['url']}")
#             response = self.session.get(filing_info['url'], timeout=60)
#             response.raise_for_status()
            
#             # Save the content
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 f.write(response.text)
            
#             logger.info(f"Successfully downloaded {file_path}")
#             time.sleep(0.2)  # Be respectful to SEC servers
#             return str(file_path)
        
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Network error downloading {company} {filing_info['year']}: {e}")
#             return None
#         except Exception as e:
#             logger.error(f"Error downloading {company} {filing_info['year']}: {e}")
#             return None
    
#     def download_all_filings(self) -> Dict[str, List[str]]:
#         """Download all required filings"""
#         downloaded = {}
        
#         for symbol, info in self.companies.items():
#             logger.info(f"Processing {symbol}...")
#             filings = self.get_filing_urls(info['cik'])
            
#             downloaded[symbol] = []
#             for filing in filings:
#                 file_path = self.download_filing(symbol, filing)
#                 if file_path:
#                     downloaded[symbol].append(file_path)
        
        
#         return downloaded
    
#     def test_api_access(self):
#         """Test API access for all companies"""
#         logger.info("Testing API access...")
#         for symbol, info in self.companies.items():
#             formatted_cik = info['cik'].zfill(10)
#             url = f"https://data.sec.gov/submissions/CIK{formatted_cik}.json"
            
#             try:
#                 logger.info(f"Testing {symbol} ({info['name']}) - CIK: {formatted_cik}")
#                 logger.info(f"URL: {url}")
                
#                 response = self.session.get(url, timeout=30)
#                 response.raise_for_status()
#                 data = response.json()
                
#                 # Check if we have filings data
#                 recent = data.get('filings', {}).get('recent', {})
#                 if recent:
#                     forms = recent.get('form', [])
#                     total_filings = len(forms)
#                     ten_k_count = sum(1 for form in forms if form == '10-K')
#                     logger.info(f"✅ {symbol}: {total_filings} total filings, {ten_k_count} 10-K filings")
#                 else:
#                     logger.warning(f"⚠️ {symbol}: No recent filings found")
                
#                 time.sleep(0.2)  # Be respectful
                
#             except requests.exceptions.RequestException as e:
#                 logger.error(f"❌ {symbol}: Network error - {e}")
#             except Exception as e:
#                 logger.error(f"❌ {symbol}: Error - {e}")
        
#         logger.info("API access test completed")

# class TextProcessor:
#     """Processes SEC filings to extract text"""
    
#     def __init__(self):
#         self.key_sections = [
#             "Item 1.", "Item 1A.", "Item 7.", "Item 8.", 
#             "Management's Discussion", "Risk Factors",
#             "Financial Statements", "Results of Operations"
#         ]
    
#     def extract_text_from_html(self, file_path: str) -> str:
#         """Extract clean text from HTML filing"""
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
            
#             soup = BeautifulSoup(content, 'html.parser')
            
#             # Remove script and style elements
#             for script in soup(["script", "style"]):
#                 script.decompose()
            
#             # Extract text
#             text = soup.get_text()
            
#             # Clean up whitespace
#             lines = (line.strip() for line in text.splitlines())
#             chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#             text = ' '.join(chunk for chunk in chunks if chunk)
            
#             return text
        
#         except Exception as e:
#             logger.error(f"Error extracting text from {file_path}: {e}")
#             return ""
    
#     def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
#         """Split text into overlapping chunks"""
#         if len(text) < chunk_size:
#             return [text]
        
#         chunks = []
#         start = 0
        
#         while start < len(text):
#             end = start + chunk_size
            
#             # Try to break at sentence boundary
#             if end < len(text):
#                 # Look for sentence ending within last 200 chars
#                 sentence_end = text.rfind('.', start + chunk_size - 200, end)
#                 if sentence_end > start:
#                     end = sentence_end + 1
            
#             chunk = text[start:end].strip()
#             if chunk:
#                 chunks.append(chunk)
            
#             start = end - overlap
        
#         return chunks

# class DataExtractionPipeline:
#     """Main pipeline for data extraction and preprocessing"""
    
#     def __init__(self, data_dir: str = "data"):
#         self.data_dir = data_dir
#         self.downloader = SECFilingDownloader(data_dir)
#         self.processor = TextProcessor()
    
#     def extract_and_process(self, force_download: bool = False) -> List[Document]:
#         """Main method to extract and process all data"""
        
#         # Step 1: Download filings
#         if force_download or not any(Path(self.data_dir).glob("*.html")):
#             logger.info("Downloading SEC filings...")
#             self.downloader.download_all_filings()
#         else:
#             logger.info("Using existing downloaded filings...")
        
#         # Step 2: Process documents
#         logger.info("Processing documents...")
#         documents = self._process_all_documents()
        
#         # Step 3: Save processed documents
#         self._save_processed_documents(documents)
        
#         return documents
    
#     def _process_all_documents(self) -> List[Document]:
#         """Process all downloaded filings into documents"""
#         documents = []
#         data_path = Path(self.data_dir)
        
#         for file_path in data_path.glob("*.html"):
#             # Extract company and year from filename
#             parts = file_path.stem.split('_')
#             if len(parts) >= 2:
#                 company = parts[0]
#                 year = parts[1]
                
#                 logger.info(f"Processing {company} {year}...")
                
#                 text = self.processor.extract_text_from_html(str(file_path))
#                 if text:
#                     chunks = self.processor.chunk_text(text)
                    
#                     for i, chunk in enumerate(chunks):
#                         doc = Document(
#                             content=chunk,
#                             company=company,
#                             year=year,
#                             filing_type="10-K",
#                             chunk_id=f"{company}_{year}_{i}"
#                         )
#                         documents.append(doc)
        
#         logger.info(f"Created {len(documents)} document chunks")
#         return documents
    
#     def _save_processed_documents(self, documents: List[Document]):
#         """Save processed documents to JSON file"""
#         output_file = Path(self.data_dir) / "processed_documents.json"
        
#         # Convert documents to serializable format
#         docs_dict = []
#         for doc in documents:
#             docs_dict.append({
#                 'content': doc.content,
#                 'company': doc.company,
#                 'year': doc.year,
#                 'filing_type': doc.filing_type,
#                 'page': doc.page,
#                 'section': doc.section,
#                 'chunk_id': doc.chunk_id
#             })
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(docs_dict, f, indent=2, ensure_ascii=False)
        
#         logger.info(f"Saved {len(documents)} processed documents to {output_file}")
    
#     def load_processed_documents(self) -> List[Document]:
#         """Load processed documents from JSON file"""
#         input_file = Path(self.data_dir) / "processed_documents.json"
        
#         if not input_file.exists():
#             logger.warning(f"No processed documents found at {input_file}")
#             return []
        
#         with open(input_file, 'r', encoding='utf-8') as f:
#             docs_dict = json.load(f)
        
#         documents = []
#         for doc_data in docs_dict:
#             doc = Document(
#                 content=doc_data['content'],
#                 company=doc_data['company'],
#                 year=doc_data['year'],
#                 filing_type=doc_data['filing_type'],
#                 page=doc_data.get('page'),
#                 section=doc_data.get('section'),
#                 chunk_id=doc_data.get('chunk_id')
#             )
#             documents.append(doc)
        
#         logger.info(f"Loaded {len(documents)} processed documents from {input_file}")
#         return documents

# def main():
#     """Main execution function for data extraction"""
    
#     # Initialize extraction pipeline
#     pipeline = DataExtractionPipeline()
    
#     # First, test API access
#     logger.info("Testing SEC API access...")
#     pipeline.downloader.test_api_access()
    
#     # Extract and process data
#     logger.info("Starting data extraction pipeline...")
#     documents = pipeline.extract_and_process(force_download=False)
    
#     # Print summary
#     companies = set(doc.company for doc in documents)
#     years = set(doc.year for doc in documents)
    
#     print("\n" + "="*50)
#     print("DATA EXTRACTION COMPLETE")
#     print("="*50)
#     print(f"Total documents processed: {len(documents)}")
#     print(f"Companies: {', '.join(sorted(companies))}")
#     print(f"Years: {', '.join(sorted(years))}")
#     print(f"Data saved to: {Path('data').absolute()}")
#     print("="*50)
    
#     # Show sample document
#     if documents:
#         sample_doc = documents[0]
#         print(f"\nSample document:")
#         print(f"Company: {sample_doc.company}")
#         print(f"Year: {sample_doc.year}")
#         print(f"Content preview: {sample_doc.content[:200]}...")
#     else:
#         print("\n⚠️ No documents were processed. Check the logs above for errors.")

# if __name__ == "__main__":
#     main()



import os
import json
import re
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from bs4 import BeautifulSoup
import time
import PyPDF2
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document chunk with metadata"""
    content: str
    company: str
    year: str
    filing_type: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None

class SECFilingDownloader:
    """Downloads 10-K filings from SEC EDGAR"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Financial RAG System 1.0; academic-research@example.com)',
            'Accept-Encoding': 'gzip, deflate',
            'Accept': 'application/json, text/html',
            'Connection': 'keep-alive'
        })
        
        # Company info with correct CIKs
        self.companies = {
            'GOOGL': {'cik': '1652044', 'name': 'Alphabet Inc.'},
            'MSFT': {'cik': '789019', 'name': 'Microsoft Corporation'},
            'NVDA': {'cik': '1045810', 'name': 'NVIDIA Corporation'}
        }
    
    def get_filing_urls(self, cik: str, form_type: str = "10-K") -> List[Dict]:
        """Get filing URLs for a company - includes both HTML and PDF links"""
        # Ensure CIK is 10 digits with leading zeros
        formatted_cik = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{formatted_cik}.json"
        
        logger.info(f"Fetching filings from: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            filings = []
            recent = data.get('filings', {}).get('recent', {})
            
            if not recent:
                logger.warning(f"No recent filings found for CIK {cik}")
                return []
            
            forms = recent.get('form', [])
            filing_dates = recent.get('filingDate', [])
            accession_numbers = recent.get('accessionNumber', [])
            primary_documents = recent.get('primaryDocument', [])
            
            for i, form in enumerate(forms):
                if form == form_type and i < len(filing_dates):
                    filing_date = filing_dates[i]
                    year = int(filing_date[:4])
                    if year in [2022, 2023, 2024]:
                        if i < len(accession_numbers) and i < len(primary_documents):
                            accession = accession_numbers[i].replace('-', '')
                            primary_doc = primary_documents[i]
                            
                            # Create both HTML and PDF URLs
                            base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"
                            html_url = f"{base_url}/{primary_doc}"
                            
                            # PDF is usually the same filename but with .pdf extension
                            # or sometimes has -filing.pdf suffix
                            pdf_filename = primary_doc.replace('.htm', '.pdf').replace('.html', '.pdf')
                            pdf_url = f"{base_url}/{pdf_filename}"
                            
                            # Alternative PDF naming conventions
                            pdf_alternatives = [
                                f"{base_url}/{primary_doc.split('.')[0]}.pdf",
                                f"{base_url}/filing.pdf",
                                f"{base_url}/{accession}.pdf"
                            ]
                            
                            filings.append({
                                'year': year,
                                'html_url': html_url,
                                'pdf_url': pdf_url,
                                'pdf_alternatives': pdf_alternatives,
                                'accession': accession,
                                'filing_date': filing_date,
                                'primary_document': primary_doc
                            })
            
            logger.info(f"Found {len(filings)} {form_type} filings for years 2022-2024")
            return sorted(filings, key=lambda x: x['year'])
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching filings for CIK {cik}: {e}")
            return []
        except KeyError as e:
            logger.error(f"Unexpected data structure for CIK {cik}: missing key {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching filings for CIK {cik}: {e}")
            return []
    
    def download_filing(self, company: str, filing_info: Dict, prefer_pdf: bool = True) -> Optional[str]:
        """Download a single filing - tries PDF first, then HTML"""
        
        if prefer_pdf:
            # Try PDF first
            pdf_path = self.data_dir / f"{company}_{filing_info['year']}_10K.pdf"
            if pdf_path.exists():
                logger.info(f"PDF file already exists: {pdf_path}")
                return str(pdf_path)
            
            # Try main PDF URL first
            if self._download_file(filing_info['pdf_url'], pdf_path):
                return str(pdf_path)
            
            # Try alternative PDF URLs
            for alt_url in filing_info.get('pdf_alternatives', []):
                if self._download_file(alt_url, pdf_path):
                    return str(pdf_path)
            
            logger.warning(f"PDF not available for {company} {filing_info['year']}, trying HTML...")
        
        # Fallback to HTML
        html_path = self.data_dir / f"{company}_{filing_info['year']}_10K.html"
        if html_path.exists():
            logger.info(f"HTML file already exists: {html_path}")
            return str(html_path)
        
        if self._download_file(filing_info['html_url'], html_path):
            return str(html_path)
        
        logger.error(f"Failed to download both PDF and HTML for {company} {filing_info['year']}")
        return None
    
    def _download_file(self, url: str, file_path: Path) -> bool:
        """Helper method to download a single file"""
        try:
            logger.info(f"Trying to download from: {url}")
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Save the content
            mode = 'wb' if file_path.suffix == '.pdf' else 'w'
            encoding = None if file_path.suffix == '.pdf' else 'utf-8'
            
            with open(file_path, mode, encoding=encoding) as f:
                if file_path.suffix == '.pdf':
                    f.write(response.content)
                else:
                    f.write(response.text)
            
            logger.info(f"✅ Successfully downloaded {file_path}")
            time.sleep(0.2)  # Be respectful to SEC servers
            return True
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"❌ Network error downloading from {url}: {e}")
            return False
        except Exception as e:
            logger.debug(f"❌ Error downloading from {url}: {e}")
            return False
    
    def download_all_filings(self) -> Dict[str, List[str]]:
        """Download all required filings"""
        downloaded = {}
        
        for symbol, info in self.companies.items():
            logger.info(f"Processing {symbol}...")
            filings = self.get_filing_urls(info['cik'])
            
            downloaded[symbol] = []
            for filing in filings:
                file_path = self.download_filing(symbol, filing)
                if file_path:
                    downloaded[symbol].append(file_path)
        
        
        return downloaded
    
    def test_api_access(self):
        """Test API access for all companies"""
        logger.info("Testing API access...")
        for symbol, info in self.companies.items():
            formatted_cik = info['cik'].zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{formatted_cik}.json"
            
            try:
                logger.info(f"Testing {symbol} ({info['name']}) - CIK: {formatted_cik}")
                logger.info(f"URL: {url}")
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Check if we have filings data
                recent = data.get('filings', {}).get('recent', {})
                if recent:
                    forms = recent.get('form', [])
                    total_filings = len(forms)
                    ten_k_count = sum(1 for form in forms if form == '10-K')
                    logger.info(f"✅ {symbol}: {total_filings} total filings, {ten_k_count} 10-K filings")
                else:
                    logger.warning(f"⚠️ {symbol}: No recent filings found")
                
                time.sleep(0.2)  # Be respectful
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ {symbol}: Network error - {e}")
            except Exception as e:
                logger.error(f"❌ {symbol}: Error - {e}")
        
        logger.info("API access test completed")

class TextProcessor:
    """Processes SEC filings to extract text"""
    
    def __init__(self):
        self.key_sections = [
            "Item 1.", "Item 1A.", "Item 7.", "Item 8.", 
            "Management's Discussion", "Risk Factors",
            "Financial Statements", "Results of Operations"
        ]
    
    def extract_text_from_html(self, file_path: str) -> str:
        """Extract clean text from HTML filing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) < chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within last 200 chars
                sentence_end = text.rfind('.', start + chunk_size - 200, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks

class DataExtractionPipeline:
    """Main pipeline for data extraction and preprocessing"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.downloader = SECFilingDownloader(data_dir)
        self.processor = TextProcessor()
    
    def extract_and_process(self, force_download: bool = False) -> List[Document]:
        """Main method to extract and process all data"""
        
        # Step 1: Download filings
        if force_download or not any(Path(self.data_dir).glob("*.html")):
            logger.info("Downloading SEC filings...")
            self.downloader.download_all_filings()
        else:
            logger.info("Using existing downloaded filings...")
        
        # Step 2: Process documents
        logger.info("Processing documents...")
        documents = self._process_all_documents()
        
        # Step 3: Save processed documents
        self._save_processed_documents(documents)
        
        return documents
    
    def _process_all_documents(self) -> List[Document]:
        """Process all downloaded filings into documents"""
        documents = []
        data_path = Path(self.data_dir)
        
        for file_path in data_path.glob("*.html"):
            # Extract company and year from filename
            parts = file_path.stem.split('_')
            if len(parts) >= 2:
                company = parts[0]
                year = parts[1]
                
                logger.info(f"Processing {company} {year}...")
                
                text = self.processor.extract_text_from_html(str(file_path))
                if text:
                    chunks = self.processor.chunk_text(text)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            content=chunk,
                            company=company,
                            year=year,
                            filing_type="10-K",
                            chunk_id=f"{company}_{year}_{i}"
                        )
                        documents.append(doc)
        
        logger.info(f"Created {len(documents)} document chunks")
        return documents
    
    def _save_processed_documents(self, documents: List[Document]):
        """Save processed documents to JSON file"""
        output_file = Path(self.data_dir) / "processed_documents.json"
        
        # Convert documents to serializable format
        docs_dict = []
        for doc in documents:
            docs_dict.append({
                'content': doc.content,
                'company': doc.company,
                'year': doc.year,
                'filing_type': doc.filing_type,
                'page': doc.page,
                'section': doc.section,
                'chunk_id': doc.chunk_id
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(documents)} processed documents to {output_file}")
    
    def load_processed_documents(self) -> List[Document]:
        """Load processed documents from JSON file"""
        input_file = Path(self.data_dir) / "processed_documents.json"
        
        if not input_file.exists():
            logger.warning(f"No processed documents found at {input_file}")
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            docs_dict = json.load(f)
        
        documents = []
        for doc_data in docs_dict:
            doc = Document(
                content=doc_data['content'],
                company=doc_data['company'],
                year=doc_data['year'],
                filing_type=doc_data['filing_type'],
                page=doc_data.get('page'),
                section=doc_data.get('section'),
                chunk_id=doc_data.get('chunk_id')
            )
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} processed documents from {input_file}")
        return documents

def main():
    """Main execution function for data extraction"""
    
    # Initialize extraction pipeline
    pipeline = DataExtractionPipeline()
    
    # First, test API access
    logger.info("Testing SEC API access...")
    pipeline.downloader.test_api_access()
    
    # Extract and process data
    logger.info("Starting data extraction pipeline...")
    documents = pipeline.extract_and_process(force_download=False)
    # print(f"Total documents processed: {documents}")
    
    # Print summary
    companies = set(doc.company for doc in documents)
    years = set(doc.year for doc in documents)
    
    print("\n" + "="*50)
    print("DATA EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total documents processed: {len(documents)}")
    print(f"Companies: {', '.join(sorted(companies))}")
    print(f"Years: {', '.join(sorted(years))}")
    print(f"Data saved to: {Path('data').absolute()}")
    print("="*50)
    
    # Show sample document
    if documents:
        sample_doc = documents[0]
        print(f"\nSample document:")
        print(f"Company: {sample_doc.company}")
        print(f"Year: {sample_doc.year}")
        print(f"Content preview: {sample_doc.content[:200]}...")
    else:
        print("\n⚠️ No documents were processed. Check the logs above for errors.")

if __name__ == "__main__":
    main()