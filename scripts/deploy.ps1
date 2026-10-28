# Reconstruye y reinicia la plataforma SÓLO si no hay entrenamientos en cola o en proceso.
#   .\scripts\deploy.ps1          # despliegue seguro
#   .\scripts\deploy.ps1 -Force   # despliega aunque haya entrenamientos (se reintentarán)
param([switch]$Force)
Set-Location (Join-Path $PSScriptRoot "..")

$sql = "select count(*) from training_jobs where status in ('pending','preparing','training','validating','exporting')"
$active = 0
$out = docker compose exec -T db psql -U vision -d vision -tAc $sql 2>$null
if ($LASTEXITCODE -eq 0 -and $out) { $active = [int](($out | Out-String).Trim()) }

if ($active -gt 0 -and -not $Force) {
    Write-Host "Hay $active entrenamiento(s) en cola o en proceso. No se despliega para no interrumpirlos."
    docker compose exec -T db psql -U vision -d vision -c "select j.id, p.name as proyecto, j.status, round(j.percent::numeric,0) as pct from training_jobs j join projects p on p.id=j.project_id where j.status in ('pending','preparing','training','validating','exporting') order by j.id"
    Write-Host "Espera a que terminen, cancelalos desde Administracion, o usa -Force."
    exit 1
}

Write-Host "Sin entrenamientos activos: construyendo y reiniciando..."
docker compose up -d --build
docker compose ps
