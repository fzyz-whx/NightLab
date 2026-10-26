param([string]$VenvPath = '.venv')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    $venv = if ([IO.Path]::IsPathRooted($VenvPath)) { $VenvPath } else { Join-Path $repoRoot $VenvPath }
    $projectPython = Join-Path $venv 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run scripts/setup_app.ps1 first (matching VenvPath).' }
    if (-not (Test-Path -LiteralPath 'frontend/dist/index.html')) { throw 'Frontend build missing. Run scripts/setup_app.ps1 first.' }
    # Probe binding without touching another process. Uvicorn also checks at bind time.
    $probe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,8766)
    $probe.Server.ExclusiveAddressUse = $true
    try { $probe.Start() } catch { throw '127.0.0.1:8766 is occupied. Close its owner yourself or use the existing app.' }
    finally { $probe.Stop() }
    $names = @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
    $previous = @{}
    foreach ($name in $names) { $previous[$name] = [Environment]::GetEnvironmentVariable($name,'Process'); [Environment]::SetEnvironmentVariable($name,'2','Process') }
    try {
        Write-Host 'NightLab: http://127.0.0.1:8766 ; Ctrl+C stops this foreground server.'
        & $projectPython -m uvicorn nightlab.api.app:create_app --factory --host 127.0.0.1 --port 8766 --workers 1
        if ($LASTEXITCODE -ne 0) { throw 'NightLab exited with an error.' }
    } finally { foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') } }
} finally { Pop-Location }
