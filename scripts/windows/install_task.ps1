<#
.SYNOPSIS
Registers the dcs2-supervisor scheduled task (#20, used by #24).

.DESCRIPTION
Registers a daily task that runs run_supervisor.ps1 in the logged-on user's session ("run only when user is logged
on"), because CS2 and HLAE need the interactive desktop. It does not run as SYSTEM and does not ask for elevated
rights. The task's execution time limit is the time from -Start to -Until plus 45 minutes, so a hung run is stopped
by Windows shortly after the deadline. There is deliberately no "start when available": if the PC was off at -Start,
that night is skipped instead of starting hours late and running into the morning. Run it from a normal PowerShell as the user who stays logged in overnight.

.PARAMETER Start
Daily start time, HH:mm (default 23:00).

.PARAMETER Until
Deadline passed to `cs2wm supervise --until`, HH:mm (default 07:30). A deadline earlier than -Start is tomorrow's.

.PARAMETER TargetHours
Stop once this many QA-passing hours are synced.

.PARAMETER Profile
Generation profile, dm or smoke.

.PARAMETER Venv
Python venv that holds cs2wm (default C:\dcs2\venv).

.PARAMETER Force
Replace the task if it is already registered.

.PARAMETER Remove
Unregister the task and exit.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\windows\install_task.ps1 -Start 23:00 -Until 07:30 -TargetHours 20 -Profile dm

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\windows\install_task.ps1 -Remove
#>
param(
    [string]$Start = "23:00",
    [string]$Until = "07:30",
    [double]$TargetHours = 20,
    [ValidateSet("dm", "smoke")][string]$Profile = "dm",
    [string]$Venv = "C:\dcs2\venv",
    [switch]$Force,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$TaskName = "dcs2-supervisor"
$Culture = [System.Globalization.CultureInfo]::InvariantCulture

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($Remove) {
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task $TaskName."
    } else {
        Write-Host "Scheduled task $TaskName is not registered."
    }
    exit 0
}

if ($existing -and -not $Force) {
    [Console]::Error.WriteLine("Scheduled task $TaskName is already registered. Use -Force to replace it, or -Remove to delete it.")
    exit 1
}

$startTime = [datetime]::ParseExact($Start, "HH:mm", $Culture)
$untilTime = [datetime]::ParseExact($Until, "HH:mm", $Culture)
$span = $untilTime - $startTime
if ($span.TotalMinutes -le 0) { $span = $span.Add([TimeSpan]::FromDays(1)) }
$limit = $span.Add([TimeSpan]::FromMinutes(45))

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$runScript = Join-Path $PSScriptRoot "run_supervisor.ps1"
$hours = $TargetHours.ToString($Culture)
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runScript`" -Until $Until -TargetHours $hours -Profile $Profile -Venv `"$Venv`""

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At $startTime
$user = "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit $limit -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

if ($existing) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false }
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
    -Description "cs2wm overnight pipeline: cs2wm supervise --until $Until --target-hours $hours --profile $Profile" | Out-Null

Write-Host "Registered scheduled task $TaskName for $user: daily at $Start, until $Until, target $hours h, profile $Profile."
Write-Host "Time limit $([int]$limit.TotalMinutes) min. The PC must stay on and $user must stay logged in."
Write-Host "The screen may lock only if the #14 locked-screen check says CS2 renders locked."
Write-Host "Start it now with: Start-ScheduledTask -TaskName $TaskName"
