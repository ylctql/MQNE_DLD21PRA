param(
    [ValidateSet('cancer','spt','mnist')][string]$Task = 'cancer',
    [ValidateSet('pilot','paper')][string]$Scale = 'pilot',
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    & $Python mqne.py --config "configs/${Task}_${Scale}.json" --output "results/${Task}_${Scale}_${taskStamp}" --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; inspect the output above.' }
} finally { Pop-Location }
