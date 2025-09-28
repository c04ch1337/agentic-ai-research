.PHONY: build run stop clean install dev test quickstart

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
	docker-compose up -d && echo "Services started! Web interface: http://localhost:8080" 