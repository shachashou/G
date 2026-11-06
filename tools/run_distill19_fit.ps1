param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('prepare-b','fit-float','fit-qat','summary-b')]
    [string]$Stage,
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunTag = 'initial'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$fitDirectory = 'data/m5/distill19/fit_b'
if ($Stage -eq 'prepare-b') {
    & $projectPython trainer/local_ranking_fit.py prepare-b --analysis-dir data/m5/distill19/analysis_initial --out-dir $fitDirectory
} elseif ($Stage -eq 'summary-b') {
    & $projectPython tools/local_ranking_summary.py --dir $fitDirectory
} else {
    $arm = $Stage.Substring(4)
    $commandArgs = @('trainer/local_ranking_fit.py','fit-b','--manifest',"$fitDirectory/fit_manifest.json",'--arm',$arm,
        '--out-dir',"$fitDirectory/${arm}_$RunTag")
    $runnerArgs = @('tools/budget_run.py','--ledger','data/m5/budget.json','--seconds','300',
        '--out-dir',"data/m5/distill19_fit_${arm}_${RunTag}_run",'--',$projectPython) + $commandArgs
    & $projectPython @runnerArgs
}
exit $LASTEXITCODE
