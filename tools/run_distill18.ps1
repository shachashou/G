param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('prepare','baselines','preflight','seal','train-d','train-s','assess-d','assess-s','summary','screen')]
    [string]$Stage
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Project .venv is missing' }
$experimentDir = 'data/m5/distill18'
$seconds = 0
$commandArgs = @()
switch ($Stage) {
    'prepare' {
        $seconds = 900
        $commandArgs = @('trainer/distill18.py','prepare','--parent-frozen','data/m5/frozen_validation.json',
            '--train-manifest','data/m5/fresh_200k_manifest.json','--probe','build/engine/Release/nnue_probe.exe',
            '--out-dir',$experimentDir,'--with-baselines')
    }
    'baselines' {
        $seconds = 120
        $commandArgs = @('trainer/distill18.py','baselines','--out-dir',$experimentDir)
    }
    'preflight' {
        $seconds = 300
        $commandArgs = @('trainer/distill18.py','preflight','--out-dir',$experimentDir)
    }
    'seal' { $commandArgs = @('trainer/distill18.py','seal','--out-dir',$experimentDir) }
    { $_ -in 'train-d','train-s' } {
        $arm = $Stage.Substring(6)
        $seconds = 600
        $commandArgs = @('trainer/train.py','--data','data/raw/bench-d4/train.binpack',
            '--frozen-validation',"$experimentDir/frozen_validation.json",'--train-manifest',"$experimentDir/train_manifest.json",
            '--target-contract',"$experimentDir/target_$arm.json",'--output',"weights/m5_distill18_$arm.pt",
            '--history',"$experimentDir/$arm/history.json",'--epochs','300','--batch-size','1024','--max-steps','1600',
            '--checkpoint-policy','last','--quant-forward','--quant-reg','0','--wdl-weight','0','--score-clip','0',
            '--lr','0.1','--hidden-lr','0.001','--seed','42')
    }
    { $_ -in 'assess-d','assess-s' } {
        $arm = $Stage.Substring(7)
        $seconds = 600
        $commandArgs = @('trainer/distill18.py','assess','--out-dir',$experimentDir,'--arm',$arm)
    }
    'summary' { $commandArgs = @('tools/distill18_summary.py','--dir',$experimentDir) }
    'screen' {
        $seconds = 300
        $commandArgs = @('tools/distill18_screen.py','--dir',$experimentDir)
    }
}
if ($seconds -gt 0) {
    $runDirectory = "data/m5/distill18_${Stage}_run"
    $runnerArgs = @('tools/budget_run.py','--ledger','data/m5/budget.json','--seconds',"$seconds",'--out-dir',$runDirectory,'--',$projectPython) + $commandArgs
    & $projectPython @runnerArgs
} else {
    & $projectPython @commandArgs
}
exit $LASTEXITCODE
