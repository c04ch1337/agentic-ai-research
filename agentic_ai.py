# Agentic AI Research Framework
# Main application structure

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import click
import chromadb
from chromadb.config import Settings
import openai
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from io import StringIO
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ResearchQuery:
    """Structured research query with reasoning steps"""
    query: str
    context: Optional[str] = None
    reasoning_steps: List[str] = None
    priority: int = 1
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.reasoning_steps is None:
            self.reasoning_steps = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ResearchResult:
    """Research result with reasoning chain"""
    query: str
    answer: str
    reasoning_chain: List[str]
    sources: List[str]
    confidence: float
    timestamp: datetime
    
class ReasoningEngine:
    """Core reasoning engine for agentic behavior"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.reasoning_history = []
        
    async def decompose_query(self, query: str) -> List[str]:
        """Break down complex queries into reasoning steps"""
        prompt = f"""
        Break down this research query into logical reasoning steps:
        Query: {query}
        
        Provide a list of specific, actionable steps to research this topic thoroughly.
        Format as a numbered list of steps.
        """
        
        try:
            response = await self._call_llm(prompt)
            steps = self._parse_steps(response)
            return steps
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return [query]  # Fallback to original query
    
    async def reason_through_step(self, step: str, context: str = "") -> str:
        """Apply reasoning to a specific step"""
        prompt = f"""
        Research Step: {step}
        Context: {context}
        
        Provide a detailed analysis and reasoning for this step.
        Consider multiple perspectives and potential implications.
        """
        
        try:
            response = await self._call_llm(prompt)
            self.reasoning_history.append({
                'step': step,
                'reasoning': response,
                'timestamp': datetime.now()
            })
            return response
        except Exception as e:
            logger.error(f"Error reasoning through step: {e}")
            return f"Unable to process step: {step}"
    
    async def synthesize_results(self, step_results: List[str], original_query: str) -> str:
        """Synthesize multiple reasoning results into final answer"""
        combined_reasoning = "\n\n".join(step_results)
        
        prompt = f"""
        Original Query: {original_query}
        
        Research Results:
        {combined_reasoning}
        
        Synthesize these results into a comprehensive, well-reasoned answer.
        Highlight key insights and connections between different aspects.
        """
        
        try:
            return await self._call_llm(prompt)
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            return "Unable to synthesize research results."
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the language model with reasoning prompt"""
        # Mock implementation - replace with actual LLM call
        # This would connect to your preferred LRM (Large Reasoning Model)
        await asyncio.sleep(0.1)  # Simulate API call
        return f"Reasoning response for: {prompt[:50]}..."
    
    def _parse_steps(self, response: str) -> List[str]:
        """Parse numbered steps from LLM response"""
        lines = response.split('\n')
        steps = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                step = line.split('.', 1)[-1].strip()
                if step:
                    steps.append(step)
        return steps if steps else [response]

class RAGSystem:
    """Retrieval-Augmented Generation system"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("research_docs")
        
    async def add_document(self, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a document to the RAG system"""
        try:
            content = await self._extract_content(file_path)
            if not content:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            # Split into chunks
            chunks = self._chunk_text(content)
            
            # Generate embeddings and add to collection
            for i, chunk in enumerate(chunks):
                doc_id = f"{Path(file_path).stem}_{i}"
                embedding = self.embedding_model.encode(chunk).tolist()
                
                doc_metadata = {
                    "source": file_path,
                    "chunk_id": i,
                    "timestamp": datetime.now().isoformat()
                }
                if metadata:
                    doc_metadata.update(metadata)
                
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[doc_metadata],
                    ids=[doc_id]
                )
            
            logger.info(f"Added {len(chunks)} chunks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    async def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )
            ]
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
                if last_period > chunk_size * 0.5:  # Only if we find a reasonable break point
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [chunk for chunk in chunks if chunk.strip()]

class AgenticAI:
    """Main agentic AI system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.reasoning_engine = ReasoningEngine()
        self.rag_system = RAGSystem()
        self.research_history = []
        
    async def research(self, query: str, use_rag: bool = True) -> ResearchResult:
        """Conduct agentic research on a query"""
        logger.info(f"Starting research on: {query}")
        
        # Decompose query into reasoning steps
        reasoning_steps = await self.reasoning_engine.decompose_query(query)
        
        # Gather context from RAG if enabled
        context = ""
        sources = []
        if use_rag:
            rag_results = await self.rag_system.search(query)
            context = "\n".join([result["content"] for result in rag_results])
            sources = list(set([result["metadata"]["source"] for result in rag_results]))
        
        # Process each reasoning step
        step_results = []
        for step in reasoning_steps:
            step_context = f"{context}\n\nPrevious reasoning: {' '.join(step_results[-2:])}"
            result = await self.reasoning_engine.reason_through_step(step, step_context)
            step_results.append(result)
        
        # Synthesize final answer
        final_answer = await self.reasoning_engine.synthesize_results(step_results, query)
        
        # Create result object
        result = ResearchResult(
            query=query,
            answer=final_answer,
            reasoning_chain=reasoning_steps,
            sources=sources,
            confidence=0.8,  # Could implement confidence scoring
            timestamp=datetime.now()
        )
        
        self.research_history.append(result)
        logger.info(f"Research completed for: {query}")
        
        return result
    
    async def add_knowledge(self, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """Add knowledge to the system"""
        return await self.rag_system.add_document(file_path, metadata)
    
    def get_research_history(self) -> List[ResearchResult]:
        """Get research history"""
        return self.research_history
    
    def save_session(self, file_path: str):
        """Save current session"""
        session_data = {
            "research_history": [asdict(result) for result in self.research_history],
            "reasoning_history": self.reasoning_engine.reasoning_history,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
    
    def load_session(self, file_path: str):
        """Load previous session"""
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            
            # Restore research history
            self.research_history = []
            for result_data in session_data.get("research_history", []):
                result_data["timestamp"] = datetime.fromisoformat(result_data["timestamp"])
                self.research_history.append(ResearchResult(**result_data))
            
            # Restore reasoning history
            self.reasoning_engine.reasoning_history = session_data.get("reasoning_history", [])
            
            logger.info(f"Session loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")

# CLI Implementation
@click.group()
@click.pass_context
def cli(ctx):
    """Agentic AI Research Framework CLI"""
    ctx.ensure_object(dict)
    ctx.obj['ai'] = AgenticAI()

@cli.command()
@click.argument('query')
@click.option('--no-rag', is_flag=True, help='Disable RAG system')
@click.option('--save', type=str, help='Save result to file')
@click.pass_context
async def research(ctx, query: str, no_rag: bool, save: str):
    """Conduct research on a query"""
    ai = ctx.obj['ai']
    
    click.echo(f"🔍 Researching: {query}")
    click.echo("=" * 50)
    
    try:
        result = await ai.research(query, use_rag=not no_rag)
        
        click.echo("📋 REASONING STEPS:")
        for i, step in enumerate(result.reasoning_chain, 1):
            click.echo(f"  {i}. {step}")
        
        click.echo("\n📚 SOURCES:")
        for source in result.sources:
            click.echo(f"  • {source}")
        
        click.echo(f"\n💡 ANSWER:")
        click.echo(result.answer)
        
        click.echo(f"\n🎯 Confidence: {result.confidence:.2f}")
        
        if save:
            with open(save, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            click.echo(f"💾 Result saved to {save}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--metadata', type=str, help='JSON metadata for the document')
@click.pass_context
async def upload(ctx, file_path: str, metadata: str):
    """Upload a document to the knowledge base"""
    ai = ctx.obj['ai']
    
    meta_dict = {}
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            click.echo("❌ Invalid JSON metadata")
            return
    
    click.echo(f"📤 Uploading: {file_path}")
    
    success = await ai.add_knowledge(file_path, meta_dict)
    
    if success:
        click.echo("✅ Document uploaded successfully")
    else:
        click.echo("❌ Failed to upload document")

@cli.command()
@click.pass_context
def history(ctx):
    """Show research history"""
    ai = ctx.obj['ai']
    
    history = ai.get_research_history()
    
    if not history:
        click.echo("📭 No research history found")
        return
    
    click.echo("📚 RESEARCH HISTORY:")
    click.echo("=" * 50)
    
    for i, result in enumerate(history, 1):
        click.echo(f"{i}. {result.query}")
        click.echo(f"   🕐 {result.timestamp}")
        click.echo(f"   🎯 Confidence: {result.confidence:.2f}")
        click.echo()

@cli.command()
@click.argument('file_path')
@click.pass_context
def save_session(ctx, file_path: str):
    """Save current session"""
    ai = ctx.obj['ai']
    ai.save_session(file_path)
    click.echo(f"💾 Session saved to {file_path}")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def load_session(ctx, file_path: str):
    """Load previous session"""
    ai = ctx.obj['ai']
    ai.load_session(file_path)
    click.echo(f"📂 Session loaded from {file_path}")

@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive mode"""
    ai = ctx.obj['ai']
    
    click.echo("🤖 Agentic AI Interactive Mode")
    click.echo("Commands: research <query>, upload <file>, history, save <file>, load <file>, quit")
    click.echo("=" * 50)
    
    while True:
        try:
            command = click.prompt("🔸", type=str).strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            # Parse command
            parts = command.split(' ', 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd == 'research' and args:
                result = asyncio.run(ai.research(args))
                click.echo(f"💡 {result.answer}")
            
            elif cmd == 'upload' and args:
                success = asyncio.run(ai.add_knowledge(args))
                click.echo("✅ Uploaded" if success else "❌ Failed")
            
            elif cmd == 'history':
                history = ai.get_research_history()
                for i, result in enumerate(history[-5:], 1):  # Show last 5
                    click.echo(f"{i}. {result.query}")
            
            elif cmd == 'save' and args:
                ai.save_session(args)
                click.echo(f"💾 Saved to {args}")
            
            elif cmd == 'load' and args:
                ai.load_session(args)
                click.echo(f"📂 Loaded from {args}")
            
            else:
                click.echo("❓ Unknown command")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            click.echo(f"❌ Error: {e}")
    
    click.echo("👋 Goodbye!")

if __name__ == "__main__":
    # Handle async commands
    import asyncio
    from functools import wraps
    
    def async_command(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Apply async wrapper to async commands
    research.callback = async_command(research.callback)
    upload.callback = async_command(upload.callback)
    
    cli()