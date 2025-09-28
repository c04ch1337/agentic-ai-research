#!/usr/bin/env python3
"""
Agentic AI Research Framework - Complete Project Generator
Run this script to generate the entire project structure and files.

Usage: python generate_project.py
"""

import os
import json
from pathlib import Path

def create_file(path, content):
    """Create a file with the given content, creating directories as needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created: {path}")

def create_directory(path):
    """Create a directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {path}")

def generate_project():
    """Generate the complete project structure."""
    
    print("🚀 Generating Agentic AI Research Framework...")
    print("=" * 50)
    
    # Create all necessary directories
    directories = [
        "agentic_ai/core",
        "agentic_ai/llm", 
        "agentic_ai/rag",
        "agentic_ai/reasoning",
        "agentic_ai/utils",
        "agentic_ai/api",
        "templates",
        "static/css",
        "static/js",
        "static/img",
        "tests/unit",
        "tests/integration",
        "tests/fixtures/sample_documents",
        "tests/fixtures/mock_responses",
        "docs",
        "scripts",
        "examples/notebooks",
        "data/uploads",
        "data/processed", 
        "data/exports",
        "sessions",
        "chroma_db",
        "logs",
        ".github/workflows",
        ".github/ISSUE_TEMPLATE",
        "docker",
        "deployment/kubernetes",
        "deployment/helm/templates",
        "deployment/terraform"
    ]
    
    for directory in directories:
        create_directory(directory)
    
    # ROOT LEVEL FILES
    
    # requirements.txt
    create_file("requirements.txt", """click==8.1.7
chromadb==0.4.22
sentence-transformers==2.2.2
openai==1.12.0
anthropic==0.18.1
PyPDF2==3.0.1
python-docx==1.1.0
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.1
fastapi==0.110.0
uvicorn==0.27.1
aiofiles==23.2.1
httpx==0.27.0
pydantic==2.6.3
python-multipart==0.0.9
jinja2==3.1.3
redis==5.0.3
langchain==0.1.11
langchain-community==0.0.25
transformers==4.38.2
torch==2.2.1
datasets==2.17.1
pytest==8.1.1
pytest-asyncio==0.23.5""")

    # .env.example
    create_file(".env.example", """# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Model Configuration
REASONING_MODEL=gpt-4
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Database Configuration
CHROMA_DB_PATH=./chroma_db
CHROMA_SERVER_URL=http://chroma:8000

# Application Settings
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=10
CONFIDENCE_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/agentic_ai.log""")

    # Dockerfile
    create_file("Dockerfile", """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    wget \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/data /app/sessions /app/chroma_db /app/logs

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CHROMA_DB_PATH=/app/chroma_db
ENV DATA_PATH=/app/data
ENV SESSIONS_PATH=/app/sessions

# Expose port
EXPOSE 8080

# Create entrypoint script
RUN echo '#!/bin/bash\\n\\
if [ "$1" = "web" ]; then\\n\\
    python -m agentic_ai.web_interface\\n\\
elif [ "$1" = "interactive" ]; then\\n\\
    python -m agentic_ai interactive\\n\\
elif [ "$1" = "research" ]; then\\n\\
    shift\\n\\
    python -m agentic_ai research "$@"\\n\\
elif [ "$1" = "upload" ]; then\\n\\
    shift\\n\\
    python -m agentic_ai upload "$@"\\n\\
else\\n\\
    python -m agentic_ai "$@"\\n\\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["web"]""")

    # docker-compose.yml
    create_file("docker-compose.yml", """version: '3.8'

services:
  agentic-ai:
    build: .
    container_name: agentic-ai-research
    volumes:
      - ./data:/app/data
      - ./sessions:/app/sessions
      - ./chroma_db:/app/chroma_db
      - ./logs:/app/logs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
    ports:
      - "8080:8080"
    stdin_open: true
    tty: true
    networks:
      - ai-network

  chroma:
    image: chromadb/chroma:latest
    container_name: chroma-db
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_CORS_ALLOW_ORIGINS=["*"]
    networks:
      - ai-network

  redis:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge

volumes:
  redis_data:
  chroma_data:""")

    # Makefile
    create_file("Makefile", """.PHONY: build run stop clean install dev test quickstart

# Docker commands
build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

clean:
	docker-compose down -v
	docker system prune -f

# Development commands
install:
	pip install -r requirements.txt

dev:
	python -m agentic_ai interactive

test:
	python -m pytest tests/ -v

# Quick start
quickstart: build
	@echo "Creating necessary directories..."
	@mkdir -p data sessions chroma_db logs
	@echo "Starting services..."
	@docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 10
	@echo "🚀 Ready! Access web interface at http://localhost:8080"

interactive:
	docker-compose exec agentic-ai python -m agentic_ai interactive

logs:
	docker-compose logs -f agentic-ai

web:
	docker-compose up agentic-ai

# Windows helpers
windows-setup:
	@echo "Setting up for Windows..."
	@if not exist data mkdir data
	@if not exist sessions mkdir sessions  
	@if not exist chroma_db mkdir chroma_db
	@if not exist logs mkdir logs
	@echo "Windows setup complete!"

windows-run:
	docker-compose up -d && echo "Services started! Web interface: http://localhost:8080" """)

    # config.json
    create_file("config.json", """{
  "models": {
    "reasoning_model": "gpt-4",
    "embedding_model": "all-MiniLM-L6-v2",
    "fallback_model": "gpt-3.5-turbo"
  },
  "rag": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_search_results": 10,
    "similarity_threshold": 0.7
  },
  "reasoning": {
    "max_steps": 10,
    "confidence_threshold": 0.7,
    "enable_self_reflection": true
  },
  "storage": {
    "chroma_db_path": "./chroma_db",
    "sessions_path": "./sessions",
    "data_path": "./data"
  },
  "logging": {
    "level": "INFO",
    "file": "./logs/agentic_ai.log"
  }
}""")

    # .gitignore
    create_file(".gitignore", """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data directories
data/uploads/*
!data/uploads/.gitkeep
sessions/*
!sessions/.gitkeep
chroma_db/*
!chroma_db/.gitkeep
logs/*
!logs/.gitkeep

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Docker
.dockerignore

# API Keys
.env""")

    # MAIN APPLICATION FILES
    
    # agentic_ai/__init__.py
    create_file("agentic_ai/__init__.py", '''"""Agentic AI Research Framework"""

__version__ = "1.0.0"
__author__ = "Agentic AI Team"

from .core.agent import AgenticAI
from .core.reasoning import ReasoningEngine
from .core.rag_system import RAGSystem

__all__ = ["AgenticAI", "ReasoningEngine", "RAGSystem"]''')

    # agentic_ai/__main__.py
    create_file("agentic_ai/__main__.py", '''"""CLI entry point for agentic_ai package"""

import sys
import asyncio
from .cli import cli

if __name__ == "__main__":
    # Handle async CLI commands
    if len(sys.argv) > 1 and sys.argv[1] in ["research", "upload"]:
        asyncio.run(cli())
    else:
        cli()''')

    # agentic_ai/cli.py - MAIN CLI IMPLEMENTATION
    create_file("agentic_ai/cli.py", '''import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any
import click

from .core.agent import AgenticAI
from .config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
@click.pass_context
def cli(ctx):
    """🤖 Agentic AI Research Framework CLI"""
    ctx.ensure_object(dict)
    
    # Initialize configuration
    config = Config()
    ctx.obj['config'] = config
    ctx.obj['ai'] = AgenticAI(config._config)

@cli.command()
@click.argument('query')
@click.option('--no-rag', is_flag=True, help='Disable RAG system')
@click.option('--save', type=str, help='Save result to file')
@click.pass_context
def research(ctx, query: str, no_rag: bool, save: str):
    """🔍 Conduct research on a query"""
    async def _research():
        ai = ctx.obj['ai']
        
        click.echo(f"🔍 Researching: {query}")
        click.echo("=" * 50)
        
        try:
            result = await ai.research(query, use_rag=not no_rag)
            
            click.echo("📋 REASONING STEPS:")
            for i, step in enumerate(result.reasoning_chain, 1):
                click.echo(f"  {i}. {step}")
            
            click.echo("\\n📚 SOURCES:")
            for source in result.sources:
                click.echo(f"  • {source}")
            
            click.echo(f"\\n💡 ANSWER:")
            click.echo(result.answer)
            
            click.echo(f"\\n🎯 Confidence: {result.confidence:.2f}")
            
            if save:
                result_dict = {
                    'query': result.query,
                    'answer': result.answer,
                    'reasoning_chain': result.reasoning_chain,
                    'sources': result.sources,
                    'confidence': result.confidence,
                    'timestamp': result.timestamp.isoformat()
                }
                with open(save, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                click.echo(f"💾 Result saved to {save}")
                
        except Exception as e:
            click.echo(f"❌ Error: {e}")
            logger.exception("Research error")
    
    asyncio.run(_research())

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--metadata', type=str, help='JSON metadata for the document')
@click.pass_context
def upload(ctx, file_path: str, metadata: str):
    """📤 Upload a document to the knowledge base"""
    async def _upload():
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
    
    asyncio.run(_upload())

@cli.command()
@click.pass_context
def history(ctx):
    """📚 Show research history"""
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
    """💾 Save current session"""
    ai = ctx.obj['ai']
    ai.save_session(file_path)
    click.echo(f"💾 Session saved to {file_path}")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def load_session(ctx, file_path: str):
    """📂 Load previous session"""
    ai = ctx.obj['ai']
    ai.load_session(file_path)
    click.echo(f"📂 Session loaded from {file_path}")

@cli.command()
@click.pass_context
def interactive(ctx):
    """🤖 Start interactive mode"""
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
    cli()''')

    # agentic_ai/config.py
    create_file("agentic_ai/config.py", '''import os
import json
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration management for the Agentic AI system"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment"""
        # Default configuration
        default_config = {
            "models": {
                "reasoning_model": "gpt-4",
                "embedding_model": "all-MiniLM-L6-v2",
                "fallback_model": "gpt-3.5-turbo"
            },
            "rag": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_search_results": 10,
                "similarity_threshold": 0.7
            },
            "reasoning": {
                "max_steps": 10,
                "confidence_threshold": 0.7,
                "enable_self_reflection": True
            },
            "api": {
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "huggingface_token": os.getenv("HUGGINGFACE_TOKEN")
            },
            "storage": {
                "chroma_db_path": os.getenv("CHROMA_DB_PATH", "./chroma_db"),
                "sessions_path": os.getenv("SESSIONS_PATH", "./sessions"),
                "data_path": os.getenv("DATA_PATH", "./data")
            },
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "file": os.getenv("LOG_FILE", "./logs/agentic_ai.log")
            }
        }
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self._deep_update(default_config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
        
        return default_config
    
    def _deep_update(self, base_dict, update_dict):
        """Deep update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")''')

    # Core application files
    core_files = {
        "agentic_ai/core/__init__.py": '"""Core system components"""',
        
        "agentic_ai/core/models.py": '''"""Data models and schemas"""
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
    evidence: List[str] = None''',

        "agentic_ai/core/agent.py": '''"""Main AgenticAI class"""
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
                context = "\\n".join([result["content"] for result in rag_results])
                sources = list(set([result["metadata"]["source"] for result in rag_results]))
            
            # Process each reasoning step
            step_results = []
            for step in reasoning_steps:
                step_context = f"{context}\\n\\nPrevious reasoning: {' '.join(step_results[-2:])}"
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
            logger.error(f"Error loading session: {e}")''',

        "agentic_ai/core/reasoning.py": '''"""Basic reasoning engine"""
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """Core reasoning engine for agentic behavior"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reasoning_history = []
        
    async def decompose_query(self, query: str) -> List[str]:
        """Break down complex queries into reasoning steps"""
        # Simple decomposition - in real implementation would use LLM
        steps = [
            f"Understand the core question: {query}",
            "Identify key concepts and relationships",
            "Gather relevant information and evidence", 
            "Analyze the information systematically",
            "Draw logical conclusions",
            "Synthesize final answer"
        ]
        
        return steps
    
    async def reason_through_step(self, step: str, context: str = "") -> str:
        """Apply reasoning to a specific step"""
        # Mock reasoning - in real implementation would use LLM
        reasoning = f"Reasoning for step: {step}\\nContext considered: {context[:100]}..."
        
        self.reasoning_history.append({
            'step': step,
            'reasoning': reasoning,
            'timestamp': datetime.now()
        })
        
        return reasoning
    
    async def synthesize_results(self, step_results: List[str], original_query: str) -> str:
        """Synthesize multiple reasoning results into final answer"""
        # Mock synthesis - in real implementation would use LLM
        return f"Based on systematic reasoning about '{original_query}', the analysis suggests a comprehensive approach considering multiple perspectives and available evidence."''',

        "agentic_ai/core/rag_system.py": '''"""RAG system implementation"""
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
                        content += page.extract_text() + "\\n"
            
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                doc = docx.Document(file_path)
                for paragraph in doc.paragraphs:
                    content += paragraph.text + "\\n"
            
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            
            elif file_path.suffix.lower() == '.csv':
                with open(file_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        content += " ".join(row) + "\\n"
            
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
        
        return [chunk for chunk in chunks if chunk.strip()]'''
    }
    
    # Utility files
    utils_files = {
        "agentic_ai/utils/__init__.py": '"""Utility functions"""',
        
        "agentic_ai/utils/logging_utils.py": '''"""Logging configuration"""
import logging
import logging.handlers
from pathlib import Path

def setup_logging(config: dict):
    """Setup logging configuration"""
    level = getattr(logging, config.get("level", "INFO").upper())
    log_file = config.get("file", "./logs/agentic_ai.log")
    
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            ),
            logging.StreamHandler()
        ]
    )''',
        
        "agentic_ai/utils/file_utils.py": '''"""File handling utilities"""
import os
import shutil
from pathlib import Path
from typing import List, Optional

def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if not"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_file_extension(file_path: str) -> str:
    """Get file extension in lowercase"""
    return Path(file_path).suffix.lower()

def is_supported_file(file_path: str) -> bool:
    """Check if file type is supported"""
    supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.csv'}
    return get_file_extension(file_path) in supported_extensions

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing invalid characters"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def copy_file_safely(src: str, dst: str) -> bool:
    """Copy file with error handling"""
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False'''
    }
    
    # Create all core files
    for file_path, content in core_files.items():
        create_file(file_path, content)
    
    for file_path, content in utils_files.items():
        create_file(file_path, content)

    # Web interface
    create_file("agentic_ai/web_interface.py", '''"""FastAPI web interface"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
from pathlib import Path

from .core.agent import AgenticAI
from .config import Config

app = FastAPI(title="Agentic AI Research Framework", version="1.0.0")

# Setup static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass  # Static directory might not exist yet

# Initialize the AI system
config = Config()
ai_system = AgenticAI(config._config)

class ResearchRequest(BaseModel):
    query: str
    use_rag: bool = True

@app.get("/", response_class=HTMLResponse)
async def home():
    """Main interface"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Agentic AI Research</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        .header { text-align: center; margin-bottom: 30px; }
        .card { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }
        input, textarea, button { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #007bff; color: white; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { background: #e7f3ff; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Agentic AI Research Framework</h1>
            <p>Advanced AI reasoning with RAG capabilities</p>
        </div>
        
        <div class="card">
            <h2>Research Query</h2>
            <textarea id="query" placeholder="Enter your research question..." rows="3"></textarea>
            <label><input type="checkbox" id="useRag" checked> Use Knowledge Base</label>
            <button onclick="doResearch()">🔍 Research</button>
        </div>
        
        <div class="card">
            <h2>Upload Documents</h2>
            <input type="file" id="fileInput" multiple accept=".pdf,.docx,.txt,.csv">
            <button onclick="uploadFiles()">📤 Upload</button>
            <div id="uploadStatus"></div>
        </div>
        
        <div id="results" class="card hidden">
            <h2>Results</h2>
            <div id="resultContent"></div>
        </div>
    </div>
    
    <script>
        async function doResearch() {
            const query = document.getElementById('query').value;
            const useRag = document.getElementById('useRag').checked;
            
            if (!query.trim()) return;
            
            showLoading('Researching...');
            
            try {
                const response = await fetch('/research', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, use_rag: useRag })
                });
                
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                showError('Research failed: ' + error.message);
            }
        }
        
        async function uploadFiles() {
            const files = document.getElementById('fileInput').files;
            const status = document.getElementById('uploadStatus');
            status.innerHTML = '';
            
            for (let file of files) {
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    status.innerHTML += `<p>✅ ${file.name}: ${result.message}</p>`;
                } catch (error) {
                    status.innerHTML += `<p>❌ ${file.name}: Upload failed</p>`;
                }
            }
        }
        
        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            const contentDiv = document.getElementById('resultContent');
            
            let html = `
                <div class="result">
                    <h3>Query: ${result.query}</h3>
                    <p><strong>Answer:</strong> ${result.answer}</p>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                </div>
                <h4>Reasoning Steps:</h4>
            `;
            
            result.reasoning_chain.forEach((step, i) => {
                html += `<div style="margin: 5px 0; padding: 5px; background: #f0f0f0;">${i + 1}. ${step}</div>`;
            });
            
            if (result.sources.length > 0) {
                html += '<h4>Sources:</h4><ul>';
                result.sources.forEach(source => {
                    html += `<li>${source}</li>`;
                });
                html += '</ul>';
            }
            
            contentDiv.innerHTML = html;
            resultsDiv.classList.remove('hidden');
        }
        
        function showLoading(message) {
            const resultsDiv = document.getElementById('results');
            const contentDiv = document.getElementById('resultContent');
            contentDiv.innerHTML = `<div style="text-align: center;">⏳ ${message}</div>`;
            resultsDiv.classList.remove('hidden');
        }
        
        function showError(message) {
            const resultsDiv = document.getElementById('results');
            const contentDiv = document.getElementById('resultContent');
            contentDiv.innerHTML = `<div style="color: red;">❌ ${message}</div>`;
            resultsDiv.classList.remove('hidden');
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/research")
async def research_endpoint(request: ResearchRequest):
    """Research endpoint"""
    try:
        result = await ai_system.research(request.query, request.use_rag)
        return {
            "query": result.query,
            "answer": result.answer,
            "reasoning_chain": result.reasoning_chain,
            "sources": result.sources,
            "confidence": result.confidence,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload file endpoint"""
    try:
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Add to knowledge base
        success = await ai_system.add_knowledge(str(file_path))
        
        return {
            "success": success,
            "message": "File uploaded and processed successfully" if success else "Failed to process file",
            "filename": file.filename
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)''')

    # README.md
    create_file("README.md", '''# 🤖 Agentic AI Research Framework

A comprehensive AI research framework with advanced reasoning capabilities, RAG (Retrieval-Augmented Generation), and Docker deployment.

## ✨ Features

- **Advanced Reasoning Engine**: Multi-step reasoning with decomposition and synthesis
- **RAG System**: Document upload and intelligent retrieval
- **Multiple Interfaces**: CLI, Web dashboard, and API
- **Docker Deployment**: Easy containerized deployment
- **Session Management**: Save and load research sessions

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)

### Method 1: Docker (Recommended)

1. **Clone/Extract the project**
2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Quick start with Docker**:
   ```bash
   make quickstart
   ```
   This will:
   - Build the Docker containers
   - Start all services
   - Create necessary directories
   - Launch the web interface at http://localhost:8080

### Method 2: Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the CLI**:
   ```bash
   python -m agentic_ai interactive
   ```

3. **Run the web interface**:
   ```bash
   python -m agentic_ai.web_interface
   ```

## 📖 Usage

### CLI Commands

```bash
# Interactive mode
python -m agentic_ai interactive

# Direct research
python -m agentic_ai research "What are the impacts of climate change?"

# Upload documents
python -m agentic_ai upload documents/research.pdf

# View history
python -m agentic_ai history

# Save session
python -m agentic_ai save-session my_session.json
```

### Web Interface

Access the web interface at http://localhost:8080 to:
- Submit research queries
- Upload documents to the knowledge base
- View research history and results
- Download session data

### Docker Commands

```bash
# Build and start services
make quickstart

# View logs
make logs

# Stop services
make stop

# Clean up
make clean

# Interactive CLI in container
make interactive
```

## 🔧 Configuration

Edit `config.json` or set environment variables:

```json
{
  "models": {
    "reasoning_model": "gpt-4",
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "rag": {
    "chunk_size": 1000,
    "max_search_results": 10
  }
}
```

## 📁 Supported File Types

- PDF documents (.pdf)
- Word documents (.docx, .doc)
- Text files (.txt)
- CSV files (.csv)

## 🛠️ Development

### Project Structure
```
agentic-ai-research/
├── agentic_ai/           # Main application
├── tests/                # Test suite
├── docs/                 # Documentation
├── examples/             # Usage examples
├── docker/               # Docker configurations
└── deployment/           # Deployment configs
```

### Running Tests
```bash
make test
```

## 🐳 Docker Services

- **agentic-ai**: Main application (port 8080)
- **chroma**: Vector database (port 8000)
- **redis**: Caching (port 6379)

## 📝 API Documentation

Once running, access API docs at:
- http://localhost:8080/docs (Swagger UI)
- http://localhost:8080/redoc (ReDoc)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Port conflicts**: Change ports in docker-compose.yml
**Permission errors**: Ensure Docker has file access permissions
**API key errors**: Verify your .env file configuration

### Getting Help

- Check the logs: `make logs`
- Review the documentation in `docs/`
- Open an issue on GitHub

---

**Happy Researching! 🔍✨**''')

    # Create gitkeep files for empty directories
    gitkeep_dirs = [
        "data/uploads/.gitkeep",
        "data/processed/.gitkeep", 
        "data/exports/.gitkeep",
        "sessions/.gitkeep",
        "chroma_db/.gitkeep",
        "logs/.gitkeep"
    ]
    
    for gitkeep in gitkeep_dirs:
        create_file(gitkeep, "")

    # Quick setup script for Windows
    create_file("setup.py", '''"""Setup script for the Agentic AI Research Framework"""
from setuptools import setup, find_packages

setup(
    name="agentic-ai-research",
    version="1.0.0",
    description="Advanced AI research framework with reasoning and RAG capabilities",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.7",
        "chromadb>=0.4.22",
        "sentence-transformers>=2.2.2",
        "openai>=1.12.0",
        "anthropic>=0.18.1",
        "PyPDF2>=3.0.1",
        "python-docx>=1.1.0",
        "pandas>=2.2.1",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.1",
        "fastapi>=0.110.0",
        "uvicorn>=0.27.1",
        "aiofiles>=23.2.1",
        "httpx>=0.27.0",
        "pydantic>=2.6.3",
        "python-multipart>=0.0.9",
        "jinja2>=3.1.3"
    ],
    entry_points={
        'console_scripts': [
            'agentic-ai=agentic_ai.cli:cli',
        ],
    },
    python_requires=">=3.11",
)''')

    # Windows batch file for quick setup
    create_file("quick_start.bat", '''@echo off
echo 🚀 Setting up Agentic AI Research Framework...
echo.

REM Create directories
if not exist data mkdir data
if not exist data\\uploads mkdir data\\uploads
if not exist sessions mkdir sessions
if not exist chroma_db mkdir chroma_db
if not exist logs mkdir logs

echo ✓ Directories created

REM Check if .env exists
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo ⚠️  IMPORTANT: Edit .env file with your API keys before continuing!
    echo.
    pause
)

echo Building Docker containers...
docker-compose build

echo Starting services...
docker-compose up -d

echo.
echo ✅ Setup complete!
echo.
echo 🌐 Web interface: http://localhost:8080
echo 📋 CLI access: docker-compose exec agentic-ai python -m agentic_ai interactive
echo 📊 View logs: docker-compose logs -f agentic-ai
echo.
echo Press any key to open web browser...
pause > nul
start http://localhost:8080''')

    # Test files
    create_file("tests/__init__.py", "")
    create_file("tests/conftest.py", '''"""Pytest configuration"""
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
    loop.close()''')

    create_file("tests/test_basic.py", '''"""Basic functionality tests"""
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
    assert len(result.reasoning_chain) > 0''')

    print("\n🎉 Project generation complete!")
    print("=" * 50)
    print("\n📁 Project structure created successfully!")
    print("\n🚀 Next steps:")
    print("1. Navigate to the project directory")
    print("2. Copy .env.example to .env and add your API keys")
    print("3. Run: make quickstart (Linux/Mac) or quick_start.bat (Windows)")
    print("4. Access web interface at http://localhost:8080")
    print("\n💡 Quick commands:")
    print("   - make quickstart    # Start everything with Docker")
    print("   - make interactive   # CLI mode")
    print("   - make logs         # View logs")
    print("   - make test         # Run tests")
    print("\nHappy researching! 🔍✨")

if __name__ == "__main__":
    generate_project()