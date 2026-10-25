param(
    [string]$LocalTorchWheel = '',
    [switch]$AllowTorchDownload
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    $projectPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run scripts/setup_cpu.ps1 first.' }
    & $projectPython -c "import sys,platform; assert sys.version_info[:2] == (3,12) and platform.machine().lower() in ('amd64','x86_64')"
    if ($LASTEXITCODE -ne 0) { throw 'CPython3.12 WindowsAMD64 required.' }
    $wheelInfo = (Get-Content -LiteralPath 'docs/DL_WHEELS.json' -Raw | ConvertFrom-Json).torch
    if (-not $LocalTorchWheel) {
        $LocalTorchWheel = Join-Path $repoRoot ('data/raw/environment/wheels/' + $wheelInfo.filename)
    }
    if (Test-Path -LiteralPath $LocalTorchWheel) {
        $wheel = Get-Item -LiteralPath $LocalTorchWheel
        if ($wheel.Length -ne $wheelInfo.bytes -or (Get-FileHash -LiteralPath $wheel.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -ne $wheelInfo.official_sha256) {
            throw 'Local wheel differs from official PyTorch size/SHA256.'
        }
        if ($wheel.Name -ne $wheelInfo.filename) { throw 'Local wheel must retain the audited wheel filename.' }
    }
    elseif (-not $AllowTorchDownload) {
        throw ('CUDA wheel is ' + $wheelInfo.bytes + ' bytes. Supply the audited local wheel, or obtain explicit download approval before -AllowTorchDownload.')
    }
    & $projectPython -m pip install --require-hashes --only-binary=:all: --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-deep-common.lock
    if ($LASTEXITCODE -ne 0) {
        & $projectPython -m pip install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements-deep-common.lock
        if ($LASTEXITCODE -ne 0) { throw 'Locked small dependencies failed.' }
    }
    if (Test-Path -LiteralPath $LocalTorchWheel) {
        & $projectPython -m pip install --no-index --no-deps --require-hashes --find-links (Split-Path (Resolve-Path -LiteralPath $LocalTorchWheel).Path -Parent) -r requirements-torch-cu128.lock
    }
    else {
        & $projectPython -m pip install --no-deps --require-hashes --only-binary=:all: --index-url https://download.pytorch.org/whl/cu128 -r requirements-torch-cu128.lock
    }
    if ($LASTEXITCODE -ne 0) { throw 'Audited CUDA PyTorch installation failed.' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency failed.' }
    & $projectPython -c "import torch; print('torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available()); print('CPU inference remains available without CUDA')"
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch runtime import failed.' }
}
finally { Pop-Location }
