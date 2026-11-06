param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunTag = 'initial',
    [string]$SampleManifest = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Project .venv is missing' }
$analysisDirectory = "data/m5/distill19/analysis_$RunTag"
$runDirectory = "data/m5/distill19_analysis_${RunTag}_run"
$commandArgs = @('trainer/local_ranking_diagnosis.py','--experiment-dir','data/m5/distill18',
    '--out-dir',$analysisDirectory,'--train-parents','256','--seed','20261119','--timeout','120','--threads','1')
if ($SampleManifest) { $commandArgs += @('--sample-manifest',$SampleManifest) }
$runnerArgs = @('tools/budget_run.py','--ledger','data/m5/budget.json','--seconds','600',
    '--out-dir',$runDirectory,'--',$projectPython) + $commandArgs
& $projectPython @runnerArgs
exit $LASTEXITCODE
