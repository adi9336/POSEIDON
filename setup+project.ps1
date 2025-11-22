# Argo Float Project Setup Script for PowerShell
# Save this as setup_project.ps1 and run it

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Argo Float Project Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# 2. Create virtual environment
Write-Host "`n[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# 3. Activate virtual environment
Write-Host "`n[3/6] Activating virtual environment..." -ForegroundColor Yellow
try {
    & .\venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "⚠ Execution policy may block activation. Trying alternative..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    & .\venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
}

# 4. Upgrade pip
Write-Host "`n[4/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✓ Pip upgraded" -ForegroundColor Green

# 5. Install dependencies
Write-Host "`n[5/6] Installing dependencies..." -ForegroundColor Yellow
$packages = @(
    "pydantic",
    "langchain",
    "langchain-openai",
    "langgraph",
    "argopy",
    "pandas",
    "python-dotenv",
    "xarray",
    "netCDF4",
    "dask",
    "requests"
)

foreach ($pkg in $packages) {
    Write-Host "  Installing $pkg..." -ForegroundColor Gray
    pip install $pkg --quiet
}
Write-Host "✓ All dependencies installed" -ForegroundColor Green

# 6. Check .env file
Write-Host "`n[6/6] Checking .env file..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✓ .env file exists" -ForegroundColor Green
    
    # Check if API key is set
    $envContent = Get-Content .env -Raw
    if ($envContent -match "OPENAI_API_KEY=.+") {
        Write-Host "✓ OPENAI_API_KEY is configured" -ForegroundColor Green
    } else {
        Write-Host "⚠ OPENAI_API_KEY not set in .env" -ForegroundColor Yellow
        Write-Host "  Please add your API key to .env file" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠ .env file not found. Creating template..." -ForegroundColor Yellow
    "OPENAI_API_KEY=your_api_key_here" | Out-File -FilePath .env -Encoding UTF8
    Write-Host "✓ .env template created. Please add your API key" -ForegroundColor Green
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Green
Write-Host "1. Add your OPENAI_API_KEY to the .env file" -ForegroundColor White
Write-Host "2. Run the workflow with:" -ForegroundColor White
Write-Host "   python src/agent/graph.py" -ForegroundColor Cyan
Write-Host "`nOr use the run script:" -ForegroundColor White
Write-Host "   .\run.ps1" -ForegroundColor Cyan

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")