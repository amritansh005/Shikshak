$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:CUDA_VISIBLE_DEVICES = "-1"
Write-Host "Starting background Ollama (CPU-only) on port 11435..." -ForegroundColor Cyan
ollama serve