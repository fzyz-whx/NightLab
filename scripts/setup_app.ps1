param([string]$VenvPath = '.venv', [string]$FrontendPath = 'frontend')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:OS -ne 'Windows_NT') {
        throw 'The wheel locks require Windows AMD64 and CPython 3.12.'
    }
    $venv = if ([IO.Path]::IsPathRooted($VenvPath)) {
        [IO.Path]::GetFullPath($VenvPath)
    } else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $VenvPath))
    }
    $projectPython = Join-Path $venv 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $venv -Parent) | Out-Null
        & py -3.12 -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 venv creation failed.' }
    }
    & $projectPython -c "import sys,platform; assert sys.version_info[:2] == (3,12) and platform.machine().lower() in ('amd64','x86_64'), 'CPython 3.12 AMD64 required'"
    if ($LASTEXITCODE -ne 0) { throw 'The venv does not match the lock platform.' }
    & $projectPython -m pip install --require-hashes --only-binary=:all: --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-api.lock
    if ($LASTEXITCODE -ne 0) {
        & $projectPython -m pip install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements-api.lock
        if ($LASTEXITCODE -ne 0) { throw 'Locked CPU/API dependency installation failed.' }
    }
    & $projectPython -m pip install --no-build-isolation --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Editable package installation failed.' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
    Push-Location $FrontendPath
    try {
        & npm.cmd ci --ignore-scripts --registry=https://registry.npmjs.org
        if ($LASTEXITCODE -ne 0) { throw 'Locked frontend dependency installation failed.' }
        foreach ($task in @('typecheck','lint','build')) {
            & npm.cmd run $task
            if ($LASTEXITCODE -ne 0) { throw "Frontend $task failed." }
        }
    } finally { Pop-Location }
    Write-Host 'CPU/API and frontend ready. No Torch, datasets, or model weights were requested.'
} finally { Pop-Location }
