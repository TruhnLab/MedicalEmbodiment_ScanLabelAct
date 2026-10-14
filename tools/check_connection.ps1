<#
.SYNOPSIS
  Checks the laptop <-> LanderPi link and (optionally) the vLLM server.

.DESCRIPTION
  Output is also saved to the project's logs\ folder. The window waits for Enter at the end
  unless -NoPause is given. Easiest way to run it: double-click check_connection.bat.

.EXAMPLE
  .\tools\check_connection.ps1
  .\tools\check_connection.ps1 -VllmUrl http://vllm.example.internal:8000
#>
param(
    [string]$RobotIp = "192.168.149.1",
    [string]$VllmUrl = "",
    [switch]$NoPause
)

function Write-Result([bool]$ok, [string]$label, [string]$hint = "") {
    if ($ok) {
        Write-Host "  [OK]   $label" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $label" -ForegroundColor Red
        if ($hint) { Write-Host "         $hint" -ForegroundColor Yellow }
    }
}

function Test-Port([string]$ip, [int]$port, [int]$timeoutMs = 1500) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync($ip, $port)
        return ($task.Wait($timeoutMs) -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

$logDir = Join-Path (Split-Path $PSScriptRoot -Parent) "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("check_{0:yyyyMMdd_HHmmss}.txt" -f (Get-Date))
Start-Transcript -Path $logFile | Out-Null

try {
    Write-Host "`n== Network adapters ==" -ForegroundColor Cyan
    $robotNet = ($RobotIp -split '\.')[0..2] -join '.'
    $addrs = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" }
    foreach ($a in $addrs) {
        Write-Host ("  {0,-35} {1}" -f $a.InterfaceAlias, $a.IPAddress)
    }
    $onRobotNet = $addrs | Where-Object { $_.IPAddress -like "$robotNet.*" }
    Write-Result ([bool]$onRobotNet) "Laptop has an address in $robotNet.x" `
        "Connect the laptop's Wi-Fi to the robot hotspot (name starts with 'HW'; password: the robot's manual)."

    Write-Host "`n== Default routes (lowest metric carries internet/VPN traffic) ==" -ForegroundColor Cyan
    $routes = @(Get-NetRoute -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue | ForEach-Object {
        $ifMetric = (Get-NetIPInterface -InterfaceIndex $_.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue).InterfaceMetric
        [pscustomobject]@{ Alias = $_.InterfaceAlias; NextHop = $_.NextHop; Metric = $_.RouteMetric + $ifMetric }
    } | Sort-Object Metric)
    foreach ($r in $routes) {
        Write-Host ("  {0,-35} via {1,-16} metric {2}" -f $r.Alias, $r.NextHop, $r.Metric)
    }
    if ($routes.Count -gt 0 -and $routes[0].NextHop -eq $RobotIp) {
        Write-Host "  Warning: internet traffic currently goes to the robot, which has no internet." -ForegroundColor Yellow
        Write-Host "  Fix (admin PowerShell): Set-NetIPInterface -InterfaceAlias '$($routes[0].Alias)' -InterfaceMetric 200" -ForegroundColor Yellow
    }

    Write-Host "`n== Robot $RobotIp ==" -ForegroundColor Cyan
    $ping = Test-Connection -ComputerName $RobotIp -Count 2 -Quiet -ErrorAction SilentlyContinue
    Write-Result $ping "Ping" "Robot not reachable. Check Wi-Fi; if the VPN is connected, it may block local traffic."
    Write-Result (Test-Port $RobotIp 22) "SSH (port 22)"
    Write-Result (Test-Port $RobotIp 9090) "rosbridge WebSocket (port 9090)" `
        "rosbridge is not running. See README.md, section 'Starting rosbridge on the robot'."
    Write-Result (Test-Port $RobotIp 8080) "web_video_server (port 8080)" `
        "No video stream server. The console can fall back to rosbridge for video (slower)."

    if ($VllmUrl) {
        Write-Host "`n== vLLM $VllmUrl ==" -ForegroundColor Cyan
        try {
            $resp = Invoke-RestMethod -Uri ($VllmUrl.TrimEnd('/') + "/v1/models") -TimeoutSec 10
            Write-Result $true "GET /v1/models"
            foreach ($m in $resp.data) { Write-Host "         model: $($m.id)" }
        } catch {
            Write-Result $false "GET /v1/models" "Error: $($_.Exception.Message). Is the VPN connected?"
        }
    }
} finally {
    Write-Host "`nLog saved to $logFile"
    Stop-Transcript | Out-Null
    if (-not $NoPause) { Read-Host "`nPress Enter to close" | Out-Null }
}
