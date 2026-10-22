<#
.SYNOPSIS
Runs `cs2wm supervise` inside the project venv and logs to $env:DCS2_ROOT\logs (#20).

.DESCRIPTION
Called by the dcs2-supervisor scheduled task. Activates C:\dcs2\venv, then runs the supervisor.

.PARAMETER Until
Deadline, HH:mm.

.PARAMETER TargetHours
Stop once this many QA-passing hours are synced.

.PARAMETER Profile
Generation profile, dm or smoke.
#>
param(
    [string]$Until = "07:30",
    [double]$TargetHours = 20,
    [ValidateSet("dm", "smoke")][string]$Profile = "dm"
)

# TODO(#20): activate C:\dcs2\venv, run `cs2wm supervise --until $Until --target-hours $TargetHours --profile $Profile`
# TODO(#20): and tee the output to $env:DCS2_ROOT\logs\supervisor-<date>.log.
throw "TODO(#20)"
