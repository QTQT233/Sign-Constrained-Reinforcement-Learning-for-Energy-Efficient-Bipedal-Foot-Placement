param(
    [Parameter(Mandatory = $true)]
    [string]$LocalRoot,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $RepoRoot "src\two_link\action_weight"
$Weights = @("0.02", "0.04", "0.06")
$Conditions = @(
    "60,30(0.33m)",
    "60,30(0.33m),0.01m",
    "70,20(0.225m)",
    "70,20,(0.225m),0.01m"
)

Copy-Item -LiteralPath (Join-Path $SourceRoot "cmt_metrics.py") `
    -Destination (Join-Path $LocalRoot "cmt_metrics.py") -Force

foreach ($Weight in $Weights) {
    foreach ($Condition in $Conditions) {
        $Relative = "action_weight=$Weight\$Condition\Whole_energy_comparison_low_dim.py"
        Copy-Item -LiteralPath (Join-Path $SourceRoot $Relative) `
            -Destination (Join-Path $LocalRoot $Relative) -Force
    }
}

$Optimized = "action_weight=0.02\60,30(0.33m)\Whole_energy_comparison_low_dim_optimized.py"
Copy-Item -LiteralPath (Join-Path $SourceRoot $Optimized) `
    -Destination (Join-Path $LocalRoot $Optimized) -Force

foreach ($Weight in $Weights) {
    $LocalFlat = Join-Path $LocalRoot "action_weight=$Weight\60,30(0.33m)"
    $Copies = @(Get-ChildItem -LiteralPath $LocalFlat -File | Where-Object {
        $_.Name -like "Whole_energy_comparison_low_dim - *.py"
    })
    if ($Copies.Count -ne 1) {
        throw "Expected one archival copy for action_weight=$Weight; found $($Copies.Count)"
    }
}

# The three archival copies intentionally differ from their canonical peers.
# Patch them in place (helper import + fusion block only) before verification;
# never overwrite an archival variant with a canonical file.

& $PythonExe (Join-Path $PSScriptRoot "verify_local_action_weight_sync.py") `
    --local-root $LocalRoot
if ($LASTEXITCODE -ne 0) {
    throw "Local action-weight verification failed with exit code $LASTEXITCODE"
}
