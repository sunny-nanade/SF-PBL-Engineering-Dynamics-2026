# DSM Frontiers Manuscript — Compile Script (Windows PowerShell)
# Run from the DSM_Frontiers_Final directory
# Requires: MiKTeX or TeX Live installed

$ErrorActionPreference = "Stop"
$tex = "DSM_Frontiers_Manuscript_Standalone"

Write-Host "`n=== DSM Frontiers Manuscript Compiler ===" -ForegroundColor Cyan
Write-Host "Working in: $(Get-Location)" -ForegroundColor Gray

# Check pdflatex exists
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Host "`nERROR: pdflatex not found." -ForegroundColor Red
    Write-Host "Install MiKTeX from https://miktex.org/download" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nStep 1/4: First pdflatex pass..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode "$tex.tex" | Out-Null

Write-Host "Step 2/4: BibTeX..." -ForegroundColor Yellow
bibtex $tex | Out-Null

Write-Host "Step 3/4: Second pdflatex pass (resolving citations)..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode "$tex.tex" | Out-Null

Write-Host "Step 4/4: Third pdflatex pass (resolving references)..." -ForegroundColor Yellow
pdflatex -interaction=nonstopmode "$tex.tex" | Out-Null

if (Test-Path "$tex.pdf") {
    $size = [math]::Round((Get-Item "$tex.pdf").Length / 1KB, 1)
    Write-Host "`n=== SUCCESS ===" -ForegroundColor Green
    Write-Host "Output: $tex.pdf  ($size KB)" -ForegroundColor Green
    Write-Host "Opening PDF..." -ForegroundColor Cyan
    Start-Process "$tex.pdf"
} else {
    Write-Host "`nERROR: PDF not generated. Check $tex.log for errors." -ForegroundColor Red
    # Show last 30 lines of log
    Get-Content "$tex.log" -Tail 30
}
