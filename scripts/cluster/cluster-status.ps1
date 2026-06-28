<#
.SYNOPSIS
    Ver los jobs SLURM activos de tu usuario en el cluster.

.DESCRIPTION
    Ejecuta squeue remotamente y muestra el estado de tus jobs.

.PARAMETER Watch
    Actualiza cada 10s (como squeue -u youruser -i 10).

.EXAMPLE
    .\scripts\cluster\cluster-status.ps1
    .\scripts\cluster\cluster-status.ps1 -Watch
#>

param(
    [switch]$Watch
)

$REMOTE_HOST = "cluster_tfg"

if ($Watch) {
    Write-Host "Monitoreando jobs (Ctrl+C para parar)..." -ForegroundColor Cyan
    Write-Host ""
    while ($true) {
        Clear-Host
        $timestamp = (Get-Date).ToString("HH:mm:ss")
        Write-Host "[$timestamp] Jobs en $REMOTE_HOST  (Ctrl+C para parar)" -ForegroundColor Cyan
        Write-Host ""
        ssh $REMOTE_HOST "squeue -u youruser --format='%.10i %.30j %.8T %.10M %.5D %R' 2>/dev/null || squeue -u youruser"
        Write-Host ""
        Start-Sleep -Seconds 10
    }
} else {
    Write-Host ""
    Write-Host "Jobs en $REMOTE_HOST" -ForegroundColor Cyan
    Write-Host ""
    ssh $REMOTE_HOST "squeue -u youruser --format='%.10i %.30j %.8T %.10M %.5D %R' 2>/dev/null || squeue -u youruser"
    Write-Host ""
}
