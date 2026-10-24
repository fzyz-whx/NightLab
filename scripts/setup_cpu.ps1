$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:OS -ne 'Windows_NT') {
        throw 'The current wheel lock targets Windows AMD64 only.'
    }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 venv creation failed.' }
    }
    $projectPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    & $projectPython -c "import sys,platform; assert sys.version_info[:2] == (3,12) and platform.machine().lower() in ('amd64','x86_64'), 'CPython 3.12 AMD64 required'"
    if ($LASTEXITCODE -ne 0) { throw 'The existing venv does not match the lock platform.' }
    & $projectPython -m pip install --require-hashes --only-binary=:all: --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-cpu.lock
    if ($LASTEXITCODE -ne 0) {
        & $projectPython -m pip install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements-cpu.lock
        if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
    }
    & $projectPython -m pip install --no-build-isolation --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Local editable package installation failed.' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
}
finally {
    Pop-Location
}
