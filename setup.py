"""Setup script for the Agentic AI Research Framework"""
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
)