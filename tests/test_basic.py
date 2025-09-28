"""Basic functionality tests"""
import pytest
from agentic_ai.core.agent import AgenticAI
from agentic_ai.config import Config

def test_config_loading():
    """Test configuration loading"""
    config = Config()
    assert config.get("models.reasoning_model") is not None
    assert config.get("rag.chunk_size") == 1000

@pytest.mark.asyncio
async def test_ai_system_init():
    """Test AI system initialization"""
    config = Config()
    ai = AgenticAI(config._config)
    assert ai is not None
    assert ai.reasoning_engine is not None
    assert ai.rag_system is not None

@pytest.mark.asyncio 
async def test_basic_research():
    """Test basic research functionality"""
    config = Config()
    ai = AgenticAI(config._config)
    
    result = await ai.research("What is artificial intelligence?", use_rag=False)
    
    assert result is not None
    assert result.query == "What is artificial intelligence?"
    assert result.answer is not None
    assert len(result.reasoning_chain) > 0