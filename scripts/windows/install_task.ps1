<#
.SYNOPSIS
Registers the dcs2-supervisor scheduled task (#20, used by #24).

.DESCRIPTION
Registers a daily task that runs run_supervisor.ps1 in the logged-on user's session ("run only when user is logged
on"), because CS2 and HLAE need the interactive desktop.

.PARAMETER Start
Daily start time, HH:mm (default 23:00).

.PARAMETER Until
Deadline passed to `cs2wm supervise --until`, HH:mm (default 07:30).

.PARAMETER TargetHours
Stop once this many QA-passing hours are synced.

.PARAMETER Profile
Generation profile, dm or smoke.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\windows\install_task.ps1 -Start 23:00 -Until 07:30 -TargetHours 20 -Profile dm
#>
param(
    [string]$Start = "23:00",
    [string]$Until = "07:30",
    [double]$TargetHours = 20,
    [ValidateSet("dm", "smoke")][string]$Profile = "dm"
)

# TODO(#20): Register-ScheduledTask dcs2-supervisor -> run_supervisor.ps1 with the parameters above.
throw "TODO(#20)"
