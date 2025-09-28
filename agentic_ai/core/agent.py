"""Main AgenticAI class"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import asdict

from .models import ResearchResult
from .reasoning import ReasoningEngine
from .rag_system import RAGSystem
from ..utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

class AgenticAI:
    """Main agentic AI system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        setup_logging(self.config.get("logging", {}))
        
        self.reasoning_engine = ReasoningEngine(self.config)
        self.rag_system = RAGSystem(self.config)
        self.research_history = []
        
    async def research(self, query: str, use_rag: bool = True) -> ResearchResult:
        """Conduct agentic research on a query"""
        logger.info(f"Starting research on: {query}")
        
        try:
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
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            # Return error result
            return ResearchResult(
                query=query,
                answer=f"Research failed: {str(e)}",
                reasoning_chain=["Error occurred during research"],
                sources=[],
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    async def add_knowledge(self, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """Add knowledge to the system"""
        return await self.rag_system.add_document(file_path, metadata)
    
    def get_research_history(self) -> List[ResearchResult]:
        """Get research history"""
        return self.research_history
    
    def save_session(self, file_path: str):
        """Save current session"""
        import json
        session_data = {
            "research_history": [asdict(result) for result in self.research_history],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
    
    def load_session(self, file_path: str):
        """Load previous session"""
        import json
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            
            # Restore research history
            self.research_history = []
            for result_data in session_data.get("research_history", []):
                result_data["timestamp"] = datetime.fromisoformat(result_data["timestamp"])
                self.research_history.append(ResearchResult(**result_data))
            
            logger.info(f"Session loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")