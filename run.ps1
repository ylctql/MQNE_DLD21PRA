param(
    [ValidateSet('cancer','spt','mnist')][string]$Task = 'cancer',
    [ValidateSet('pilot','paper','fixed')][string]$Scale = 'fixed',
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $configName = if ($Scale -eq 'pilot') { "${Task}_pilot.json" }
                  elseif ($Task -eq 'mnist') { 'mnist_yao_paper.json' }
                  else { "${Task}_fixed_paper.json" }
    & $Python mqne.py --config "configs/$configName" --output "results/${Task}_${Scale}_${taskStamp}" --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; inspect the output above.' }
} finally { Pop-Location }
