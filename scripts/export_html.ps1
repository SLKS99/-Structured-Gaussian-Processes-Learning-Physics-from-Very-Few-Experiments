$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$notebook = "notebooks\sGP_Tutorial_Bandgap_Perovskites.ipynb"
$outDir = "docs"
$outName = "sgp_tutorial.html"

if (-Not (Test-Path $notebook)) {
  throw "Notebook not found at '$notebook'"
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "Exporting notebook to HTML..."
jupyter nbconvert --to html "$notebook" --output "$outName" --output-dir "$outDir"

Write-Host "Wrote:" (Join-Path $outDir $outName)

