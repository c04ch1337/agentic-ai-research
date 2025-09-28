# 🤖 Agentic AI Research Framework

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

**Happy Researching! 🔍✨**