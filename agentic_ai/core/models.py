"""Data models and schemas"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Any
from enum import Enum

@dataclass
class ResearchQuery:
    """Structured research query"""
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

class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive" 
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"

@dataclass
class ReasoningStep:
    step_type: ReasoningType
    premise: str
    reasoning: str
    conclusion: str
    confidence: float
    evidence: List[str] = None