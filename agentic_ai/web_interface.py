"""FastAPI web interface"""
from fastapi import FastAPI, UploadFile, File, HTTPException
import logging, json, os
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

@app.get("/health")
def health():
    return {"status": "ok"}

# Optional: JSON logs
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"t":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s","name":"%(name)s"}',
)

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
    uvicorn.run(app, host="0.0.0.0", port=8080)