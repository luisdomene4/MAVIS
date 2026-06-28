<#
.SYNOPSIS
    Descarga las bases de datos de embeddings (.db) del cluster a local.

.DESCRIPTION
    cluster-pull.ps1 IGNORA los .db a proposito (solo trae csv/json/png/logs).
    Este script sincroniza los SQLite de embeddings, que cambian en el cluster
    (regeneraciones de WAVE, GE2 incrementales, nuevos datasets) y son la fuente
    de verdad para el notebook de analisis (experiments/analysis/).

    Descarga (rsync incremental, solo lo que cambio):
      experiments/**/results/**/*.db   (todas las DBs de todos los datasets/modelos)

    Por defecto descarga TODO. Usa -Dataset para limitar a uno.

    NOTA: las DBs pesan (~5 GB el set de M3A). La primera vez tarda; despues
    rsync solo trae los bytes modificados (--modify-window=1).

.PARAMETER Dataset
    Limita la descarga a un dataset: FakeVV_testset | GroundLie360 | M3A.
    Si se omite, descarga los tres.

.PARAMETER DryRun
    Muestra que se descargaria sin transferir (rsync --dry-run).

.EXAMPLE
    .\scripts\cluster\cluster-pull-dbs.ps1
    .\scripts\cluster\cluster-pull-dbs.ps1 -Dataset M3A
    .\scripts\cluster\cluster-pull-dbs.ps1 -DryRun
#>

param(
    [ValidateSet("FakeVV_testset", "GroundLie360", "M3A")]
    [string]$Dataset,
    [switch]$DryRun
)

# -- Configuracion (igual que cluster-pull.ps1) -------------------
$REMOTE_HOST = "cluster_tfg"
$REMOTE_DIR  = "/home/youruser/mavis-TFG"
$REPO_ROOT   = $PSScriptRoot | Split-Path | Split-Path

$drive = $REPO_ROOT.Substring(0, 1).ToLower()
$pathWithoutDrive = $REPO_ROOT.Substring(2) -replace "\\", "/"
$wslLocal = "/mnt/$drive$pathWithoutDrive"

# Subdir a sincronizar: un dataset concreto o todo experiments/
$subPath = if ($Dataset) { "experiments/$Dataset" } else { "experiments" }

Write-Host ""
Write-Host "PULL DBs <- ${REMOTE_HOST}:${REMOTE_DIR}/$subPath/" -ForegroundColor Cyan
if ($DryRun) { Write-Host "  (dry-run: no se transfiere nada)" -ForegroundColor Yellow }
Write-Host ""

$start = Get-Date

$rsyncArgs = @(
    "-rtvz",
    "--modify-window=1",
    "--progress",
    "--include=*/",      # recorrer subdirectorios
    "--include=*.db",    # solo ficheros .db
    "--exclude=*",       # ignorar todo lo demas
    "${REMOTE_HOST}:${REMOTE_DIR}/$subPath/",
    "$wslLocal/$subPath/"
)
if ($DryRun) { $rsyncArgs = @("--dry-run") + $rsyncArgs }

wsl rsync @rsyncArgs
$ok = ($LASTEXITCODE -eq 0)

# -- Resumen ------------------------------------------------------
$elapsed = ((Get-Date) - $start).TotalSeconds
Write-Host ""
if ($ok) {
    Write-Host "Pull DBs completado en $([math]::Round($elapsed, 1))s" -ForegroundColor Green
} else {
    Write-Host "WARNING: Pull DBs completado con errores ($([math]::Round($elapsed, 1))s)" -ForegroundColor Yellow
}
