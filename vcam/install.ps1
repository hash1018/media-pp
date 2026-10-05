# Installs (or with -Uninstall, removes) the media-pp virtual camera's DLL.
#
# Run from an elevated PowerShell after `cargo build --release -p media-pp-vcam`:
#
#   powershell -ExecutionPolicy Bypass -File vcam\install.ps1
#   powershell -ExecutionPolicy Bypass -File vcam\install.ps1 -Uninstall
#
# The DLL is copied out of the build directory because Windows loads it into
# its Frame Server service, whose account cannot read most of a user's
# profile; and it is registered for the machine, the only place that service
# looks. Frame Server is stopped first, since it keeps a loaded DLL open and a
# new copy could not replace it. It starts again by itself when an
# application next opens a camera.
param(
    [switch]$Uninstall,
    [string]$Dll = "$PSScriptRoot\..\target\release\media_pp_vcam.dll"
)
$ErrorActionPreference = "Stop"

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this from an elevated (administrator) PowerShell."
}

# regsvr32 is a windowed program: PowerShell neither waits for it nor sets
# $LASTEXITCODE when it is called directly, so it is started and waited on.
function Invoke-Regsvr32([string[]]$Arguments) {
    $process = Start-Process regsvr32.exe -ArgumentList $Arguments -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "regsvr32 $($Arguments -join ' ') failed ($($process.ExitCode))"
    }
}

$target = Join-Path $env:ProgramFiles "media-pp\vcam"
$installed = Join-Path $target "media_pp_vcam.dll"

Stop-Service FrameServer -Force -ErrorAction SilentlyContinue
Stop-Service FrameServerMonitor -Force -ErrorAction SilentlyContinue

if ($Uninstall) {
    if (Test-Path $installed) {
        Invoke-Regsvr32 @("/s", "/u", "`"$installed`"")
        Remove-Item $target -Recurse -Force
    }
    "media-pp virtual camera removed"
    return
}

if (-not (Test-Path $Dll)) {
    throw "No DLL at $Dll - build it first: cargo build --release -p media-pp-vcam"
}
New-Item -ItemType Directory -Force $target | Out-Null
Copy-Item $Dll $installed -Force
Invoke-Regsvr32 @("/s", "`"$installed`"")
"media-pp virtual camera installed: $installed"
