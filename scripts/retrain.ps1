param([int]$Trials = 6, [int]$Iterations = 400)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
& $pythonExe -m cricket_ai.cli prepare
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
& $pythonExe -m pytest -q tests/test_pipeline.py
if ($LASTEXITCODE -ne 0) { throw 'Data correctness tests failed' }
& $pythonExe -m cricket_ai.cli train --trials $Trials --iterations $Iterations
if ($LASTEXITCODE -ne 0) { throw 'Training/evaluation failed' }
& $pythonExe scripts/render_report.py
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
Write-Output 'Training complete. Inspect reports/EVALUATION.md before explicitly promoting the new version.'
