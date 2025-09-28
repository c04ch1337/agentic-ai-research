"""Pytest configuration"""
import pytest
import asyncio
from agentic_ai.config import Config
from agentic_ai.core.agent import AgenticAI

@pytest.fixture
def config():
    """Test configuration"""
    return Config()

@pytest.fixture
async def ai_system(config):
    """Test AI system"""
    return AgenticAI(config._config)

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()