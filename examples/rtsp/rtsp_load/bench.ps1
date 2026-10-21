# Measures an RtspServer under load: starts the `rtsp_server` example on the
# loopback with `-Video`, then for each number of `-Players` runs `rtsp_load`
# for `-Seconds`, sampling the server process's processor time and memory as
# it plays. Prints one line of the server's figures and `rtsp_load`'s summary
# for each. Build both first: cargo build --release -p rtsp_server -p rtsp_load
# See docs/benchmarks/streaming/method.md.
param(
    [Parameter(Mandatory = $true)][string]$Video,
    [int[]]$Players = @(1, 50, 100, 200, 400),
    [int]$Seconds = 30,
    [int]$Threads = 2,
    [switch]$Udp,
    [int]$Port = 18600
)
$ErrorActionPreference = "Stop"
$root = Join-Path $PSScriptRoot "..\..\..\target\release"
$work = Join-Path ([System.IO.Path]::GetTempPath()) "rtsp_load_bench"
New-Item -ItemType Directory -Force $work | Out-Null
$video = (Resolve-Path $Video).Path
$server = Start-Process -FilePath "$root\rtsp_server.exe" -WorkingDirectory $work -PassThru -WindowStyle Hidden `
    -ArgumentList @("`"$video`"", "127.0.0.1:$Port", "--seconds", "100000", "--threads", "$Threads") `
    -RedirectStandardOutput "$work\server.out" -RedirectStandardError "$work\server.err"
try {
    Start-Sleep -Seconds 3
    $url = "rtsp://127.0.0.1:$Port/file"
    foreach ($n in $Players) {
        $cpu0 = (Get-Process -Id $server.Id).TotalProcessorTime.TotalSeconds
        $t0 = Get-Date
        $peak = 0
        $loadCpu = 0
        $arguments = @($url, "$n", "--seconds", "$Seconds")
        if ($Udp) { $arguments += "--udp" }
        $load = Start-Process -FilePath "$root\rtsp_load.exe" -WorkingDirectory $work -PassThru -WindowStyle Hidden `
            -ArgumentList $arguments -RedirectStandardOutput "$work\load-$n.out" -RedirectStandardError "$work\load-$n.err"
        while (-not $load.HasExited) {
            Start-Sleep -Milliseconds 500
            $memory = (Get-Process -Id $server.Id).WorkingSet64
            if ($memory -gt $peak) { $peak = $memory }
            try { $loadCpu = (Get-Process -Id $load.Id).TotalProcessorTime.TotalSeconds } catch {}
        }
        $cpu1 = (Get-Process -Id $server.Id).TotalProcessorTime.TotalSeconds
        $wall = ((Get-Date) - $t0).TotalSeconds
        "=== players=$n threads=$Threads udp=$Udp server_cores=$([math]::Round(($cpu1 - $cpu0) / $wall, 2)) server_peak_MB=$([math]::Round($peak / 1MB, 0)) players_cores=$([math]::Round($loadCpu / $wall, 2))"
        Get-Content "$work\load-$n.out" | Select-Object -Last 4
        Start-Sleep -Seconds 3
    }
}
finally {
    Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
}
