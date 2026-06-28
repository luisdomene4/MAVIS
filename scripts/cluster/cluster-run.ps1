<#
.SYNOPSIS
    Push codigo + submit job SLURM + esperar resultado + pull output automatico.

.DESCRIPTION
    Todo el ciclo en un comando:
      1. Sube el codigo (cluster-push)
      2. Ejecuta sbatch en el cluster
      3. Monitorea el job hasta que termina
      4. Descarga el output automaticamente

.PARAMETER Sbatch
    Ruta al archivo .sbatch relativa al repo (ej: scripts/slurm/run_qwen3vl.sbatch)

.PARAMETER NoPush
    Salta el push de codigo (asume que ya esta actualizado).

.PARAMETER PollInterval
    Segundos entre comprobaciones de estado (default: 20).

.EXAMPLE
    .\scripts\cluster\cluster-run.ps1 scripts/slurm/run_qwen3vl.sbatch
    .\scripts\cluster\cluster-run.ps1 scripts/slurm/test_gpu.sbatch -NoPush
    .\scripts\cluster\cluster-run.ps1 scripts/slurm/run_qwen3vl.sbatch -PollInterval 30
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Sbatch,

    [switch]$NoPush,

    [int]$PollInterval = 20
)

$REMOTE_HOST = "cluster_tfg"
$REMOTE_DIR  = "/home/youruser/mavis-TFG"
$REPO_ROOT   = $PSScriptRoot | Split-Path | Split-Path

Write-Host ""
Write-Host "cluster-run: $Sbatch" -ForegroundColor Cyan
Write-Host ""

# -- 1. Push codigo ------------------------------------------------
if (-not $NoPush) {
    Write-Host "-- [1/3] Push codigo -------------------------------" -ForegroundColor DarkCyan
    & "$PSScriptRoot\cluster-push.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Push fallo, abortando." -ForegroundColor Red
        exit 1
    }
    Write-Host ""
} else {
    Write-Host "-- [1/3] Push omitido (-NoPush)" -ForegroundColor DarkGray
}

# -- 2. Submit sbatch ----------------------------------------------
Write-Host "-- [2/3] Submitting $Sbatch ------------------------" -ForegroundColor DarkCyan
$sbatchOutput = ssh $REMOTE_HOST "cd '$REMOTE_DIR' && sbatch $Sbatch" 2>&1
Write-Host "   $sbatchOutput"

# Extraer Job ID ("Submitted batch job 4800")
if ($sbatchOutput -match "Submitted batch job (\d+)") {
    $jobId = $Matches[1]
    Write-Host "   Job ID: $jobId" -ForegroundColor Green
} else {
    Write-Host "No se pudo extraer el Job ID. Fallo el sbatch?" -ForegroundColor Red
    Write-Host $sbatchOutput
    exit 1
}

# -- 3. Monitorear hasta que termine ------------------------------
Write-Host ""
Write-Host "-- [3/3] Monitoreando job $jobId (intervalo: ${PollInterval}s) --" -ForegroundColor DarkCyan
Write-Host "   Ctrl+C para cancelar la espera (el job seguira corriendo en el cluster)"
Write-Host ""

$startTime = Get-Date

while ($true) {
    Start-Sleep -Seconds $PollInterval

    $status = ssh $REMOTE_HOST "squeue -j $jobId -h -o '%T' 2>/dev/null" 2>$null
    $status = if ($status) { $status.Trim() } else { "" }

    $elapsed = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)
    $timestamp = (Get-Date).ToString("HH:mm:ss")

    if ([string]::IsNullOrEmpty($status)) {
        # Job ya no esta en la cola -> termino
        Write-Host "  [$timestamp] Job $jobId - COMPLETADO (${elapsed}min)" -ForegroundColor Green
        break
    } elseif ($status -match "FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY") {
        Write-Host "  [$timestamp] Job $jobId - $status (${elapsed}min)" -ForegroundColor Yellow
        break
    } else {
        # Mostrar las ultimas 3 lineas del .out si existe
        $lastLines = ssh $REMOTE_HOST "tail -3 '$REMOTE_DIR/logs/*_${jobId}.out' 2>/dev/null" 2>$null
        Write-Host "  [$timestamp] $status (${elapsed}min)" -ForegroundColor Gray
        if ($lastLines) {
            $lastLines | ForEach-Object { Write-Host "    | $_" -ForegroundColor DarkGray }
        }
    }
}

# -- 4. Pull outputs -----------------------------------------------
Write-Host ""
Write-Host "-- [4/4] Descargando outputs -----------------------" -ForegroundColor DarkCyan
& "$PSScriptRoot\cluster-pull.ps1"

# Mostrar el output final del job
Write-Host ""
$outFile = Get-ChildItem "$REPO_ROOT\logs" -Filter "*_${jobId}.out" 2>$null | Select-Object -First 1
$errFile = Get-ChildItem "$REPO_ROOT\logs" -Filter "*_${jobId}.err" 2>$null | Select-Object -First 1

if ($outFile) {
    Write-Host "Output: $($outFile.FullName)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "-- Ultimas 20 lineas del .out ----------------------" -ForegroundColor DarkCyan
    Get-Content $outFile.FullName | Select-Object -Last 20
}
if ($errFile -and (Get-Item $errFile.FullName).Length -gt 0) {
    Write-Host ""
    Write-Host "-- .err (ultimas 10 lineas) ------------------------" -ForegroundColor Yellow
    Get-Content $errFile.FullName | Select-Object -Last 10
}

Write-Host ""
$totalMin = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)
Write-Host "Ciclo completo en ${totalMin}min" -ForegroundColor Green
