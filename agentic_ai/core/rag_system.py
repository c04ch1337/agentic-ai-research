"""RAG system implementation"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
import PyPDF2
import docx
import csv
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGSystem:
    """Retrieval-Augmented Generation system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documents = []  # Simple in-memory storage
        
    async def add_document(self, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a document to the RAG system"""
        try:
            content = await self._extract_content(file_path)
            if not content:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            # Simple chunking
            chunks = self._chunk_text(content)
            
            # Store chunks with metadata
            for i, chunk in enumerate(chunks):
                doc_entry = {
                    "content": chunk,
                    "metadata": {
                        "source": file_path,
                        "chunk_id": i,
                        "timestamp": datetime.now().isoformat(),
                        **(metadata or {})
                    }
                }
                self.documents.append(doc_entry)
            
            logger.info(f"Added {len(chunks)} chunks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    async def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents - simple keyword matching"""
        try:
            query_lower = query.lower()
            results = []
            
            for doc in self.documents:
                content_lower = doc["content"].lower()
                # Simple relevance score based on keyword overlap
                score = sum(1 for word in query_lower.split() if word in content_lower)
                
                if score > 0:
                    results.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": score
                    })
            
            # Sort by relevance and return top results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    async def _extract_content(self, file_path: str) -> str:
        """Extract text content from various file types"""
        file_path = Path(file_path)
        content = ""
        
        try:
            if file_path.suffix.lower() == '.pdf':
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
            
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                doc = docx.Document(file_path)
                for paragraph in doc.paragraphs:
                    content += paragraph.text + "\n"
            
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            
            elif file_path.suffix.lower() == '.csv':
                with open(file_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        content += " ".join(row) + "\n"
            
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
        
        return content.strip()
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [chunk for chunk in chunks if chunk.strip()]