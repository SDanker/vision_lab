# Genera .env con secretos aleatorios a partir de .env.example (no sobrescribe un .env existente).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path .env) { Write-Host ".env ya existe; no se modifica."; exit 0 }

function New-Secret([int]$len) {
    $chars = [char[]]"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    $bytes = New-Object byte[] $len
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
}

# Clave Fernet: 32 bytes aleatorios en base64 url-safe
function New-FernetKey {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    [Convert]::ToBase64String($bytes).Replace("+", "-").Replace("/", "_")
}

$text = Get-Content .env.example -Raw
$text = $text.Replace("CHANGE_ME_POSTGRES_PASSWORD", (New-Secret 32)).Replace("CHANGE_ME_SECRET_KEY_AT_LEAST_32_CHARS", (New-Secret 64)).Replace("CHANGE_ME_HF_ENCRYPTION_KEY", (New-FernetKey))
[System.IO.File]::WriteAllText((Join-Path (Get-Location) ".env"), $text, (New-Object System.Text.UTF8Encoding($false)))
Write-Host ".env creado con secretos aleatorios."
Write-Host "Siguiente paso: docker compose up -d --build"
Write-Host "Luego crea el administrador: docker compose exec api python -m app.cli create-admin --username admin --email tu@correo.cl"
