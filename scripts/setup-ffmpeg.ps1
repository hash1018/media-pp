<#
.SYNOPSIS
Builds the FFmpeg media-pp is built against, and points the build at it.

.DESCRIPTION
media-pp needs FFmpeg 8.0's development files: its bindings break on 8.1's
headers, and prebuilt 8.0 trees are no longer published. This builds 8.0.1
from vcpkg at the tag CI uses (the same one .github/actions/setup-ffmpeg
pins), into a folder of your own, and sets FFMPEG_DIR and PATH:

  - FFMPEG_DIR is where the build finds the headers and import libraries;
    ffmpeg-sys-next reads it before looking anywhere else.
  - PATH must hold its `bin`, or every program built against it fails to
    start with STATUS_DLL_NOT_FOUND.

Building FFmpeg takes tens of minutes the first time. It needs git and the
Visual Studio Build Tools' C++ workload; vcpkg fetches everything else.

By default the build carries OpenH264 (the always-available H.264 encoder)
and nvcodec (NVIDIA's decode and encode, loaded at run time, so harmless
without an NVIDIA GPU). -Vulkan adds what the crate's `vulkan` feature builds
against.

.PARAMETER Into
Where vcpkg and the built tree go. Default: %LOCALAPPDATA%\media-pp\ffmpeg.

.PARAMETER Vulkan
Also build FFmpeg's Vulkan support, for media-pp's `vulkan` feature.

.PARAMETER Features
The vcpkg ffmpeg features to build, in place of the defaults above.

.PARAMETER Persist
Also set FFMPEG_DIR and add the DLLs to PATH for your user account, so new
terminals and editors find them. Without it they are set for this session
only, and the lines to set them again are printed.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\setup-ffmpeg.ps1 -Persist
#>
param(
    [string]$Into = (Join-Path $env:LOCALAPPDATA "media-pp\ffmpeg"),
    [switch]$Vulkan,
    [string]$Features,
    [switch]$Persist
)
$ErrorActionPreference = "Stop"

# Keep in step with .github/actions/setup-ffmpeg/action.yml.
$VcpkgRef = "2026.01.16"
$PortVersion = "8.0.1"
$Triplet = "x64-windows"

if (-not $Features) {
    $Features = if ($Vulkan) { "openh264,nvcodec,vulkan" } else { "openh264,nvcodec" }
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is needed to fetch vcpkg: install it from https://git-scm.com and run this again."
}

$vcpkg = Join-Path $Into "vcpkg"
if (-not (Test-Path (Join-Path $vcpkg ".git"))) {
    Write-Host "Fetching vcpkg $VcpkgRef into $vcpkg"
    New-Item -ItemType Directory -Force $Into | Out-Null
    git clone --quiet --depth 1 --branch $VcpkgRef https://github.com/microsoft/vcpkg.git $vcpkg
    if ($LASTEXITCODE -ne 0) { throw "git clone of vcpkg failed ($LASTEXITCODE)" }
}

# A vcpkg already there may be at another tag, whose ffmpeg port builds
# another FFmpeg: refused rather than built from.
$port = Get-Content (Join-Path $vcpkg "ports\ffmpeg\vcpkg.json") -Raw | ConvertFrom-Json
if ($port.version -ne $PortVersion) {
    throw "the vcpkg at $vcpkg builds FFmpeg $($port.version), not $PortVersion. Remove it, or pass another -Into."
}

if (-not (Test-Path (Join-Path $vcpkg "vcpkg.exe"))) {
    & (Join-Path $vcpkg "bootstrap-vcpkg.bat") -disableMetrics
    if ($LASTEXITCODE -ne 0) { throw "vcpkg bootstrap failed ($LASTEXITCODE)" }
}
Write-Host "Building ffmpeg[$Features] — this takes a while the first time."
& (Join-Path $vcpkg "vcpkg.exe") install "ffmpeg[$Features]:$Triplet"
if ($LASTEXITCODE -ne 0) { throw "vcpkg install ffmpeg failed ($LASTEXITCODE)" }
# Nothing links the debug variant, and it is half the tree.
Remove-Item (Join-Path $vcpkg "installed\$Triplet\debug") -Recurse -Force -ErrorAction SilentlyContinue

$prefix = Join-Path $vcpkg "installed\$Triplet"
$version = Join-Path $prefix "include\libavcodec\version_major.h"
if (-not (Test-Path $version)) { throw "no FFmpeg headers under $prefix" }
if (-not (Select-String -Path $version -Pattern '#define\s+LIBAVCODEC_VERSION_MAJOR\s+62\b' -Quiet)) {
    throw "the headers under $prefix are not FFmpeg 8.0's"
}
$bin = Join-Path $prefix "bin"

$env:FFMPEG_DIR = $prefix
if (($env:PATH -split ';') -notcontains $bin) { $env:PATH = "$bin;$env:PATH" }
if ($Persist) {
    [Environment]::SetEnvironmentVariable("FFMPEG_DIR", $prefix, "User")
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if (($userPath -split ';') -notcontains $bin) {
        [Environment]::SetEnvironmentVariable("PATH", "$bin;$userPath", "User")
    }
    Write-Host "Set FFMPEG_DIR and PATH for your account: open a new terminal to pick them up."
} else {
    Write-Host "Set for this session. To set them again elsewhere:"
    Write-Host "  `$env:FFMPEG_DIR = `"$prefix`""
    Write-Host "  `$env:PATH = `"$bin;`$env:PATH`""
}
Write-Host "FFmpeg $PortVersion is ready. Check the machine with: cargo run -p doctor"
