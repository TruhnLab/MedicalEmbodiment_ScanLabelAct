# Opens a LanderPi console page; starts the local helper (tools/robot_helper.py) first unless it already
# runs with the current code. Used by start_mapping_console.bat and start_task_console.bat, so opening
# the second console doesn't restart the helper (that would lose track of running reconstructions and
# labelling jobs), while a helper with older code is still replaced as before.
param([string]$Page = "")

$repo = Split-Path -Parent $PSScriptRoot
$url = "http://127.0.0.1:8765/"

# the helper's code version: newest change of robot_helper.py and landerpi/*.py (robot_helper.code_version)
$files = @(Get-Item "$repo\tools\robot_helper.py") + @(Get-ChildItem "$repo\landerpi\*.py")
$version = ($files | ForEach-Object { [int][Math]::Floor(([DateTimeOffset]$_.LastWriteTimeUtc).ToUnixTimeMilliseconds() / 1000) } |
            Measure-Object -Maximum).Maximum

$running = $null
try { $running = (Invoke-RestMethod "$url`api/version" -TimeoutSec 2).version } catch { }

if ($running -ne $version) {
    # conda run inside cmd /k: the window stays open with the helper's output
    Start-Process cmd -ArgumentList "/k", "title LanderPi helper && cd /d `"$repo`" && conda run --no-capture-output -n landerpi python -u tools\robot_helper.py"
    # wait until it answers (the first start takes a few seconds)
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try { Invoke-RestMethod "$url`api/version" -TimeoutSec 1 | Out-Null; break } catch { }
    }
}
Start-Process "$url$Page"
