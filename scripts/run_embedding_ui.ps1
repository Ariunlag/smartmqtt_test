$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project virtual environment is missing. Create .venv and install requirements.txt first."
}

. (Join-Path $projectRoot ".venv\Scripts\Activate.ps1")
$tempPath = Join-Path $projectRoot ".tmp"
New-Item -ItemType Directory -Force -Path $tempPath | Out-Null
$env:TEMP = (Resolve-Path -LiteralPath $tempPath).Path
$env:TMP = $env:TEMP

python -m streamlit run app/embedding_explorer.py --server.fileWatcherType none
