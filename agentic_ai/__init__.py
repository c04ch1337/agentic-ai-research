"""Agentic AI Research Framework"""

__version__ = "1.0.0"
__author__ = "Agentic AI Team"

from .core.agent import AgenticAI
from .core.reasoning import ReasoningEngine
from .core.rag_system import RAGSystem

__all__ = ["AgenticAI", "ReasoningEngine", "RAGSystem"]