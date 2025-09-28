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

# Usage examples
research:
	docker-compose exec agentic-ai python -m agentic_ai research "$(QUERY)"

upload:
	docker-compose exec agentic-ai python -m agentic_ai upload "/app/uploads/$(FILE)"

# Quick start
quickstart: build
	@echo "Creating necessary directories..."
	@mkdir -p data sessions chroma_db uploads logs
	@echo "Starting services..."
	@docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 5
	@echo "Ready! Run 'make interactive' to start"

interactive:
	docker-compose exec agentic-ai python -m agentic_ai interactive

# Logs
logs:
	docker-compose logs -f agentic-ai

# Backup
backup:
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data sessions chroma_db

restore:
	@echo "Usage: make restore BACKUP=backup_file.tar.gz"
	@if [ -n "$(BACKUP)" ]; then tar -xzf $(BACKUP); fi