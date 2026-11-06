param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$previousPath = 'data/m5/distill18_train-d_run/run.json'
$retryDirectory = 'data/m5/distill18_train-d_retry1_run'
$previous = Get-Content -Raw -LiteralPath $previousPath | ConvertFrom-Json
if ($previous.status -ne 'interrupted' -or $previous.exit_code -ne 130) {
    throw 'Expected the interrupted first D training attempt (exit 130)'
}
if ((Test-Path -LiteralPath 'weights/m5_distill18_d.pt') -or
    (Test-Path -LiteralPath 'data/m5/distill18/d/history.json')) {
    throw 'Training artifacts exist; inspect them before restarting'
}
if (Test-Path -LiteralPath $retryDirectory) { throw 'Retry already attempted; refusing to overwrite' }
if (Test-Path -LiteralPath 'data/m5/budget.json.lock') { throw 'Budget locked; inspect the active job' }
$activeTraining = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(trainer[/\\]train\.py|tools[/\\]budget_run\.py)'
}
if ($activeTraining) { throw 'A training/budget process is active; refusing a duplicate run' }
$trainingCommand = @($previous.command)
if ($trainingCommand.Count -lt 3 -or
    [IO.Path]::GetFullPath($trainingCommand[0]) -ne [IO.Path]::GetFullPath($projectPython) -or
    $trainingCommand[1] -ne 'trainer/train.py' -or
    $trainingCommand[[Array]::IndexOf($trainingCommand, '--output') + 1] -ne 'weights/m5_distill18_d.pt') {
    throw 'Previous command is not the expected project D training command'
}
& $projectPython -c "import sys; from pathlib import Path; sys.path.insert(0,'trainer'); import distill18; distill18.verify_plan(Path('data/m5/distill18')); print('Frozen plan verified; original interrupted log and budget are preserved.')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) {
    Write-Output "Recovery ready: unchanged training command, new run directory $retryDirectory, 600-second limit."
    exit 0
}
$runnerArgs = @('tools/budget_run.py','--ledger','data/m5/budget.json','--seconds','600',
    '--out-dir',$retryDirectory,'--') + $trainingCommand
& $projectPython @runnerArgs
exit $LASTEXITCODE
