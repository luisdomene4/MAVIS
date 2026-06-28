<#
.SYNOPSIS
    Sube el codigo al cluster via rsync (WSL) - solo archivos de codigo, sin datos pesados.

.DESCRIPTION
    Usa rsync via WSL para sincronizacion incremental: solo transfiere lo que ha cambiado.
    Excluye automaticamente: data/, src/models/, .git/, pesos, videos, caches.

    Tiempo estimado: 2-5 segundos (tras el primer push).

.PARAMETER DryRun
    Simula sin hacer cambios. Muestra que se transferiria.

.EXAMPLE
    .\scripts\cluster\cluster-push.ps1
    .\scripts\cluster\cluster-push.ps1 -DryRun
#>

param(
    [switch]$DryRun
)

# -- Configuracion ------------------------------------------------
$REMOTE_HOST = "cluster_tfg"
$REMOTE_DIR  = "/home/youruser/mavis-TFG"
$REPO_ROOT   = $PSScriptRoot | Split-Path | Split-Path

# Convertir ruta Windows a ruta WSL (/mnt/c/...)
$drive = $REPO_ROOT.Substring(0, 1).ToLower()
$pathWithoutDrive = $REPO_ROOT.Substring(2) -replace "\\", "/"
$wslPath = "/mnt/$drive$pathWithoutDrive"

$excludeFile = "$REPO_ROOT\scripts\cluster\rsync-exclude.txt"
$wslExclude = "/mnt/$drive" + ($excludeFile.Substring(2) -replace "\\", "/")

# -- Construccion del comando rsync -------------------------------
$rsyncArgs = @(
    "-rtvz",
    "--modify-window=1",
    "--progress",
    "--exclude-from=$wslExclude"
)

if ($DryRun) {
    $rsyncArgs += "--dry-run"
    Write-Host "WARNING: DRY RUN - no se haran cambios reales" -ForegroundColor Yellow
}

$rsyncArgs += @(
    "$wslPath/",
    "${REMOTE_HOST}:${REMOTE_DIR}/"
)

# -- Ejecutar -----------------------------------------------------
Write-Host ""
Write-Host "PUSH -> ${REMOTE_HOST}:${REMOTE_DIR}" -ForegroundColor Cyan
Write-Host "   Desde : $REPO_ROOT" -ForegroundColor Gray
if ($DryRun) {
    Write-Host "   Modo  : DRY RUN" -ForegroundColor Yellow
}
Write-Host ""

$start = Get-Date
wsl rsync @rsyncArgs
$elapsed = ((Get-Date) - $start).TotalSeconds

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Push completado en $([math]::Round($elapsed, 1))s" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "rsync fallo (exit code $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
