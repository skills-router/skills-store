$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ScriptDir '.venv'
$PythonBin = Join-Path $VenvDir 'Scripts\python.exe'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error 'uv not found in PATH'
  exit 1
}

if (-not (Test-Path $VenvDir)) {
  Write-Host "Creating virtual environment: $VenvDir" 
  uv venv $VenvDir
}

Write-Host 'Installing redact runtime dependencies...'
uv pip install `
  --python "$PythonBin" `
  pillow `
  pymupdf `
  paddleocr `
  paddlepaddle

Write-Output $VenvDir
