<#
.SYNOPSIS
    Streaming en tiempo real del output de un archivo de log del cluster.

.DESCRIPTION
    Ejecuta `tail -f` sobre un archivo remoto y lo muestra en tu terminal local.
    Ideal para ver el progreso de un job mientras se ejecuta.

    Ctrl+C para parar.

.PARAMETER File
    Ruta relativa al repo del archivo a seguir (ej: logs/qwen3vl_embed_youruser_4800.out)
    Si no se especifica, busca el .out mas reciente en logs/.

.PARAMETER JobId
    ID del job SLURM para buscar automaticamente su .out/.err.

.PARAMETER Err
    Si se usa con -JobId, muestra el .err en vez del .out.

.EXAMPLE
    .\scripts\cluster\cluster-tail.ps1 logs/qwen3vl_embed_youruser_4800.out
    .\scripts\cluster\cluster-tail.ps1 -JobId 4800
    .\scripts\cluster\cluster-tail.ps1 -JobId 4800 -Err
    .\scripts\cluster\cluster-tail.ps1   # ultimo .out de logs/
#>

param(
    [Parameter(Position=0)]
    [string]$File,

    [string]$JobId,

    [switch]$Err
)

$REMOTE_HOST = "cluster_tfg"
$REMOTE_DIR  = "/home/youruser/mavis-TFG"

# -- Resolver que archivo ver --------------------------------------
if ($JobId) {
    $ext = if ($Err) { "err" } else { "out" }
    $remoteFile = ssh $REMOTE_HOST "ls '$REMOTE_DIR/logs/'*_${JobId}.${ext} 2>/dev/null | head -1" 2>$null
    if (-not $remoteFile) {
        Write-Host "No se encontro logs/*_${JobId}.${ext} en el cluster." -ForegroundColor Red
        exit 1
    }
} elseif ($File) {
    $remoteFile = "$REMOTE_DIR/$($File -replace '\\', '/')"
} else {
    # Ultimo .out de logs/
    $remoteFile = ssh $REMOTE_HOST "ls -t '$REMOTE_DIR/logs/'*.out 2>/dev/null | head -1" 2>$null
    if (-not $remoteFile) {
        Write-Host "No hay archivos .out en $REMOTE_DIR/logs/" -ForegroundColor Red
        exit 1
    }
}

$remoteFile = $remoteFile.Trim()
Write-Host ""
Write-Host "Streaming: $REMOTE_HOST:$remoteFile" -ForegroundColor Cyan
Write-Host "   Ctrl+C para parar." -ForegroundColor DarkGray
Write-Host ""

ssh $REMOTE_HOST "tail -f '$remoteFile'"
