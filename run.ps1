# Run script for Argo Float Project
# Save this as run.ps1

Write-Host "`n🌊 Starting Argo Float Workflow...`n" -ForegroundColor Cyan

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & .\venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated`n" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to activate venv. Make sure it exists!" -ForegroundColor Red
    Write-Host "  Run: python -m venv venv`n" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Set Python path to include project root
$env:PYTHONPATH = $PWD

# Check if .env file exists and has API key
if (Test-Path ".env") {
    $envContent = Get-Content .env -Raw
    if ($envContent -notmatch "OPENAI_API_KEY=sk-") {
        Write-Host "⚠ WARNING: OPENAI_API_KEY may not be set correctly in .env" -ForegroundColor Yellow
        Write-Host "  Please check your .env file`n" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠ WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host "  Create it with: OPENAI_API_KEY=your_key_here`n" -ForegroundColor Yellow
}

# First, test imports
Write-Host "Testing imports..." -ForegroundColor Yellow
python test_imports.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Import test failed!" -ForegroundColor Red
    Write-Host "  Fix the errors above before running the workflow`n" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the workflow
Write-Host "`nRunning workflow...`n" -ForegroundColor Green
python src/agent/graph.py

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Workflow completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n✗ Workflow failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}

# Keep window open
Write-Host "`nPress Enter to exit..." -ForegroundColor Gray
Read-Host