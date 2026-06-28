<#
.SYNOPSIS
    Vigila el cluster cada N segundos y descarga outputs nuevos automaticamente.

.DESCRIPTION
    Polling periodico al cluster. Cuando detecta archivos .out/.err nuevos en logs/,
    los descarga automaticamente a tu maquina local.

    Ejecuta en una terminal secundaria mientras trabajas.
    Ctrl+C para parar.

.PARAMETER Interval
    Segundos entre comprobaciones (default: 30).

.EXAMPLE
    .\scripts\cluster\cluster-watch.ps1
    .\scripts\cluster\cluster-watch.ps1 -Interval 15
#>

param(
    [int]$Interval = 30
)

$REMOTE_HOST = "cluster_tfg"
$REMOTE_DIR  = "/home/youruser/mavis-TFG"
$REPO_ROOT   = $PSScriptRoot | Split-Path | Split-Path
$drive = $REPO_ROOT.Substring(0, 1).ToLower()
$pathWithoutDrive = $REPO_ROOT.Substring(2) -replace "\\", "/"
$wslLocal = "/mnt/$drive$pathWithoutDrive"

# Archivo marcador en el cluster para detectar cambios
$MARKER = "$REMOTE_DIR/.watch_marker"

Write-Host ""
Write-Host "cluster-watch activo (intervalo: ${Interval}s)" -ForegroundColor Cyan
Write-Host "   Cluster : ${REMOTE_HOST}:${REMOTE_DIR}/logs/" -ForegroundColor Gray
Write-Host "   Local   : $REPO_ROOT\logs\" -ForegroundColor Gray
Write-Host "   Ctrl+C para parar." -ForegroundColor DarkGray
Write-Host ""

# Inicializar el marcador en el cluster
ssh $REMOTE_HOST "touch '$MARKER'" 2>$null

while ($true) {
    $timestamp = (Get-Date).ToString("HH:mm:ss")

    # Buscar archivos mas nuevos que el marcador
    $newFiles = ssh $REMOTE_HOST @"
find '$REMOTE_DIR/logs' -newer '$MARKER' \( -name '*.out' -o -name '*.err' \) 2>/dev/null
"@ 2>$null

    if ($newFiles) {
        Write-Host "[$timestamp] Nuevos outputs detectados:" -ForegroundColor Green
        $newFiles | ForEach-Object { Write-Host "   - $(Split-Path $_ -Leaf)" -ForegroundColor White }

        # Descargar logs/
        wsl rsync -avz --progress `
            --include='*/' `
            --include='*.out' `
            --include='*.err' `
            --exclude='*' `
            "${REMOTE_HOST}:${REMOTE_DIR}/logs/" `
            "$wslLocal/logs/" 2>$null

        # Actualizar el marcador
        ssh $REMOTE_HOST "touch '$MARKER'" 2>$null

        Write-Host "[$timestamp] Descargado -> $REPO_ROOT\logs\" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host "[$timestamp] Sin cambios..." -ForegroundColor DarkGray
    }

    Start-Sleep -Seconds $Interval
}
