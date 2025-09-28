@echo off
echo 🚀 Setting up Agentic AI Research Framework...
echo.

REM Create directories
if not exist data mkdir data
if not exist data\uploads mkdir data\uploads
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
start http://localhost:8080