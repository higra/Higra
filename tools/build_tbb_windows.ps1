$ErrorActionPreference = "Stop"

$TBB_VERSION = "2023.1.0"
$TBB_URL = "https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v$TBB_VERSION.tar.gz"
$TBB_INSTALL_PREFIX = "C:\tbb"

Write-Host "Building oneTBB $TBB_VERSION -> $TBB_INSTALL_PREFIX"

Invoke-WebRequest -Uri $TBB_URL -OutFile archive.tgz
tar -xzf archive.tgz
$extracted = Get-ChildItem -Directory -Filter "oneTBB-*" | Select-Object -First 1
Rename-Item -Path $extracted.FullName -NewName tbb

$cmakeConfigureArgs = @(
  "-S", "tbb",
  "-B", "tbb\build",
  "-DTBB_TEST=OFF",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_INSTALL_LIBDIR=lib",
  "-DCMAKE_INSTALL_PREFIX=$TBB_INSTALL_PREFIX"
)
Write-Host "cmake configure args: $($cmakeConfigureArgs -join ' ')"
& cmake @cmakeConfigureArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed with exit code $LASTEXITCODE" }

& cmake --build tbb\build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "cmake build failed with exit code $LASTEXITCODE" }

& cmake --install tbb\build --config Release
if ($LASTEXITCODE -ne 0) { throw "cmake install failed with exit code $LASTEXITCODE" }

if (-not (Test-Path "$TBB_INSTALL_PREFIX\bin")) {
    throw "Expected install directory $TBB_INSTALL_PREFIX\bin does not exist"
}