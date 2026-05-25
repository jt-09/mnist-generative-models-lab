# Build reports/report.pdf from the reports/ directory.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

pdflatex -interaction=nonstopmode report.tex | Out-Null
bibtex report | Out-Null
pdflatex -interaction=nonstopmode report.tex | Out-Null
pdflatex -interaction=nonstopmode report.tex | Out-Null

if (-not (Test-Path report.pdf)) {
    throw "report.pdf was not created. See report.log for details."
}

Write-Host "Built report.pdf ($(Get-Item report.pdf | Select-Object -ExpandProperty Length) bytes)"
