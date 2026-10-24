<#
.SYNOPSIS
Runs `cs2wm supervise` inside the project venv and logs to $env:DCS2_ROOT\logs (#20).

.DESCRIPTION
Called by the dcs2-supervisor scheduled task (or by hand for a short test). Activates the venv (default
C:\dcs2\venv), works from the Windows clone of the repo, runs the supervisor and tees its output to
$env:DCS2_ROOT\logs\supervisor-<date>.log. The exit code of `cs2wm supervise` is the exit code of this script:
0 normal, 1 error, 3 the night ended on a guard trip or an alert.

.PARAMETER Until
Deadline, HH:mm (the next such time, so 07:30 at 23:00 is tomorrow).

.PARAMETER TargetHours
Stop once this many QA-passing hours are synced.

.PARAMETER Profile
Generation profile, dm or smoke.

.PARAMETER Venv
Python venv that holds cs2wm (default C:\dcs2\venv).

.PARAMETER Repo
The Windows clone of the repo (default: the repo this script is in).

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\windows\run_supervisor.ps1 -Until 07:30 -TargetHours 20 -Profile dm
#>
param(
    [string]$Until = "07:30",
    [double]$TargetHours = 20,
    [ValidateSet("dm", "smoke")][string]$Profile = "dm",
    [string]$Venv = "C:\dcs2\venv",
    [string]$Repo = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
)

$ErrorActionPreference = "Stop"
if (-not $env:DCS2_ROOT) {
    [Console]::Error.WriteLine("DCS2_ROOT is not set (see issue #2).")
    exit 2
}

$logDir = Join-Path $env:DCS2_ROOT "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("supervisor-{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))
$hours = $TargetHours.ToString([System.Globalization.CultureInfo]::InvariantCulture)

Set-Location $Repo
. (Join-Path $Venv "Scripts\Activate.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

# Windows PowerShell 5.1 would write UTF-16 with Tee-Object, so the log is written with Out-File -Encoding utf8
# (UTF-8 with a BOM on 5.1). Output is also echoed to the console for a manual run.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
function Write-Log([string]$Line) {
    $Line | Out-File -FilePath $log -Append -Encoding utf8
    Write-Host $Line
}

Write-Log "$(Get-Date -Format s) run_supervisor: cs2wm supervise --until $Until --target-hours $hours --profile $Profile"

# cmd.exe merges stderr (where the Python logging goes) into stdout, so PowerShell 5.1 does not wrap it as errors.
& cmd.exe /c "cs2wm supervise --until $Until --target-hours $hours --profile $Profile 2>&1" | ForEach-Object { Write-Log $_ }
$code = $LASTEXITCODE

Write-Log "$(Get-Date -Format s) run_supervisor: exit code $code"
exit $code
