<#
 One-command OCR evaluation.
 Examples:
   .\run_demo.ps1                                   # quick demo: 3 pages per booklet, all 4 models
   .\run_demo.ps1 -Pdfs "C:\my\pdfs" -Pages 0       # your own PDFs, every page
   .\run_demo.ps1 -Models glm-ocr,qwen3vl-2b        # only some models
   .\run_demo.ps1 -Typed                            # also run the 7 typed samples (with accuracy scores)
#>
param(
  [string]$Pdfs   = "N:\tarnDEV\model_evaluation\data\sample_collected",
  [string[]]$Models = @("qwen3vl-2b","qwen3vl-4b","glm-ocr","paddleocr-vl"),
  [int]$Pages = 3,          # pages per booklet, 0 = all
  [switch]$SkipSplit,       # reuse page images already in data\raw
  [switch]$Typed
)
$ErrorActionPreference = "Stop"
Set-Location N:\tarnDEV\model_evaluation
.\.venv\Scripts\Activate.ps1

if (-not $SkipSplit) {
  Write-Host "`n[1/4] Splitting PDFs into page images..." -ForegroundColor Cyan
  python -I src/profile_pages.py $Pdfs (Get-Location).Path
}
Write-Host "`n[2/4] Building page list..." -ForegroundColor Cyan
python src/make_manifest.py data/manifest_demo.csv $Pages

if ($Typed) { Get-Content data/manifest_typed.csv | Select-Object -Skip 1 | Add-Content data/manifest_demo.csv }
Write-Host "`n[3/4] Running models one at a time (slow part)..." -ForegroundColor Cyan
foreach ($m in $Models) {
  Write-Host "--- $m ---" -ForegroundColor Yellow
  python src/run_eval.py data/manifest_demo.csv $m
}
Write-Host "`n[4/4] Summarising..." -ForegroundColor Cyan
python src/demo_summary.py @Models
Write-Host "`nDone. Open results\summary.md (table) and results\compare.md (text side by side)." -ForegroundColor Green
