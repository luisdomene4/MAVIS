<#
.SYNOPSIS
    Descarga outputs del cluster (.out, .err, CSVs, PNGs de resultados).

.DESCRIPTION
    Descarga selectivamente solo archivos de resultados/logs desde el cluster.
    NO descarga datos pesados, modelos ni videos.

    Descarga:
      - logs/*.out, logs/*.err  (outputs de SLURM)
      - experiments/**/results/*.csv, *.json, *.png (resultados)

    Tiempo estimado: 2-5 segundos.

.PARAMETER All
    Incluye tambien cambios de codigo del cluster (rsync bidireccional).

.EXAMPLE
    .\scripts\cluster\cluster-pull.ps1
    .\scripts\cluster\cluster-pull.ps1 -All
#>

param(
    [switch]$All
)

# -- Configuracion ------------------------------------------------
$REMOTE_HOST = "cluster_tfg"
$REMOTE_DIR  = "/home/youruser/mavis-TFG"
$REPO_ROOT   = $PSScriptRoot | Split-Path | Split-Path

$drive = $REPO_ROOT.Substring(0, 1).ToLower()
$pathWithoutDrive = $REPO_ROOT.Substring(2) -replace "\\", "/"
$wslLocal = "/mnt/$drive$pathWithoutDrive"

Write-Host ""
Write-Host "PULL <- ${REMOTE_HOST}:${REMOTE_DIR}" -ForegroundColor Cyan
Write-Host ""

$start = Get-Date
$anyError = $false

# -- 1. Descargar logs/ (todos los .out y .err) -------------------
Write-Host "  [1/2] Descargando logs/ (.out, .err)..." -ForegroundColor Gray

$logsArgs = @(
    "-rtvz",
    "--modify-window=1",
    "--progress",
    "--include=*/",
    "--include=*.out",
    "--include=*.err",
    "--include=*.log",
    "--exclude=*",
    "${REMOTE_HOST}:${REMOTE_DIR}/logs/",
    "$wslLocal/logs/"
)
wsl rsync @logsArgs

if ($LASTEXITCODE -ne 0) { $anyError = $true }

# -- 2. Descargar resultados de experiments/ (CSV, JSON, PNG) -----
Write-Host ""
Write-Host "  [2/2] Descargando experiments/**/results/ (csv, json, png)..." -ForegroundColor Gray

$resultsArgs = @(
    "-rtvz",
    "--modify-window=1",
    "--progress",
    "--include=*/",
    "--include=*.csv",
    "--include=*.json",
    "--include=*.png",
    "--include=*.txt",
    "--exclude=*",
    "${REMOTE_HOST}:${REMOTE_DIR}/experiments/",
    "$wslLocal/experiments/"
)
wsl rsync @resultsArgs

if ($LASTEXITCODE -ne 0) { $anyError = $true }

# -- 3. (Opcional) Sync de codigo modificado en el cluster --------
if ($All) {
    Write-Host ""
    Write-Host "  [3/3] Descargando cambios de codigo del cluster..." -ForegroundColor Gray

    $excludeFile = "$REPO_ROOT\scripts\cluster\rsync-exclude.txt"
    $wslExclude = "/mnt/$drive" + ($excludeFile.Substring(2) -replace "\\", "/")

    $codeArgs = @(
        "-rtvz",
        "--modify-window=1",
        "--progress",
        "--exclude-from=$wslExclude",
        "${REMOTE_HOST}:${REMOTE_DIR}/src/",
        "$wslLocal/src/"
    )
    wsl rsync @codeArgs
    if ($LASTEXITCODE -ne 0) { $anyError = $true }
}

# -- Resumen ------------------------------------------------------
$elapsed = ((Get-Date) - $start).TotalSeconds
Write-Host ""
if (-not $anyError) {
    Write-Host "Pull completado en $([math]::Round($elapsed, 1))s" -ForegroundColor Green
    Write-Host "   Logs en: $REPO_ROOT\logs\" -ForegroundColor Gray
} else {
    Write-Host "WARNING: Pull completado con errores ($([math]::Round($elapsed, 1))s)" -ForegroundColor Yellow
}
