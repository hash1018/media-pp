# Measures a WebRtcServer under load: starts the `webrtc_server` example on
# the loopback with `-Video`, then for each number of `-Players` runs
# `webrtc_load` for `-Seconds`, sampling the server process's processor time
# and memory as it plays. Prints one line of the server's figures and
# `webrtc_load`'s summary for each. With -Record DIR the server also records
# what it serves into an HLS event playlist there, and is left to end by
# itself after the runs, so that the playlist is finished. Build both first:
# cargo build --release -p webrtc_server -p webrtc_load
# See docs/benchmarks/streaming/method.md.
param(
    [Parameter(Mandatory = $true)][string]$Video,
    [int[]]$Players = @(1, 25, 50, 100),
    [int]$Seconds = 30,
    [int]$Threads = 2,
    [int]$UdpPort = 18700,
    [int]$HttpPort = 18701,
    [string]$Record = ""
)
$ErrorActionPreference = "Stop"
$root = Join-Path $PSScriptRoot "..\..\..\target\release"
$work = Join-Path ([System.IO.Path]::GetTempPath()) "webrtc_load_bench"
New-Item -ItemType Directory -Force $work | Out-Null
$video = (Resolve-Path $Video).Path
$serverSeconds = 100000
$serverArguments = @("`"$video`"", "--udp", "127.0.0.1:$UdpPort", "--http", "127.0.0.1:$HttpPort", "--threads", "$Threads")
if ($Record) {
    $serverSeconds = 3 + $Players.Count * ($Seconds + 15)
    $serverArguments += @("--record", "`"$Record`"")
}
$serverArguments += @("--seconds", "$serverSeconds")
$server = Start-Process -FilePath "$root\webrtc_server.exe" -WorkingDirectory $work -PassThru -WindowStyle Hidden `
    -ArgumentList $serverArguments `
    -RedirectStandardOutput "$work\server.out" -RedirectStandardError "$work\server.err"
try {
    Start-Sleep -Seconds 3
    $url = "http://127.0.0.1:$HttpPort/file/whep"
    foreach ($n in $Players) {
        $cpu0 = (Get-Process -Id $server.Id).TotalProcessorTime.TotalSeconds
        $t0 = Get-Date
        $peak = 0
        $loadCpu = 0
        $load = Start-Process -FilePath "$root\webrtc_load.exe" -WorkingDirectory $work -PassThru -WindowStyle Hidden `
            -ArgumentList @($url, "$n", "--seconds", "$Seconds") `
            -RedirectStandardOutput "$work\load-$n.out" -RedirectStandardError "$work\load-$n.err"
        while (-not $load.HasExited) {
            Start-Sleep -Milliseconds 500
            $memory = (Get-Process -Id $server.Id).WorkingSet64
            if ($memory -gt $peak) { $peak = $memory }
            try { $loadCpu = (Get-Process -Id $load.Id).TotalProcessorTime.TotalSeconds } catch {}
        }
        $cpu1 = (Get-Process -Id $server.Id).TotalProcessorTime.TotalSeconds
        $wall = ((Get-Date) - $t0).TotalSeconds
        "=== players=$n threads=$Threads server_cores=$([math]::Round(($cpu1 - $cpu0) / $wall, 2)) server_peak_MB=$([math]::Round($peak / 1MB, 0)) players_cores=$([math]::Round($loadCpu / $wall, 2))"
        Get-Content "$work\load-$n.out" | Select-Object -Last 5
        Start-Sleep -Seconds 5
    }
}
finally {
    if ($Record) {
        # Let it end by itself, finishing the recording's playlist.
        Wait-Process -Id $server.Id -Timeout ($serverSeconds + 30) -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
}
