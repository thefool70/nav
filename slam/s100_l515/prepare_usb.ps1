[CmdletBinding()]
param(
    [string]$BindBusIds = "",
    [string]$S100BusId = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$L515DescriptionPattern = "(?i)(RealSense.*L?515|L?515.*RealSense)"
$S100DescriptionPattern = "(?i)(CH9102|USB[- ]?Enhanced[- ]?SERIAL)"


function Get-UsbipdExecutable {
    $command = Get-Command usbipd.exe -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "usbipd-win is not installed or usbipd.exe is not in PATH."
    }
    return $command.Source
}


function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}


function Invoke-Usbipd {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $script:UsbipdExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "usbipd $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
}


function Get-UsbipdState {
    $jsonLines = & $script:UsbipdExecutable state
    if ($LASTEXITCODE -ne 0) {
        throw "usbipd state failed with exit code $LASTEXITCODE."
    }
    try {
        return (($jsonLines -join [Environment]::NewLine) | ConvertFrom-Json)
    }
    catch {
        throw "Cannot parse the JSON returned by usbipd state: $($_.Exception.Message)"
    }
}


function Show-ConnectedUsbDevices {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Devices
    )

    Write-Host "Connected USB devices reported by usbipd:"
    foreach ($device in $Devices) {
        if ($null -eq $device.BusId) {
            continue
        }
        $bound = $null -ne $device.PersistedGuid
        $attached = $null -ne $device.ClientIPAddress
        Write-Host (
            "  {0,-6} bound={1,-5} attached={2,-5} {3}" -f
            $device.BusId, $bound, $attached, $device.Description
        )
    }
}


function Get-ConnectedDevice {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Devices,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$DescriptionPattern,
        [string]$BusId = ""
    )

    $candidateDevices = @(
        $Devices | Where-Object {
            ($null -ne $_.BusId) -and
            ([string]$_.Description -match $DescriptionPattern)
        }
    )
    if (-not [string]::IsNullOrWhiteSpace($BusId)) {
        if ($BusId -notmatch "^[0-9]+-[0-9]+(?:\.[0-9]+)*$") {
            throw "Invalid USB bus ID '$BusId'."
        }
        $candidateDevices = @(
            $candidateDevices | Where-Object { $_.BusId -eq $BusId }
        )
    }
    if ($candidateDevices.Count -ne 1) {
        Show-ConnectedUsbDevices -Devices $Devices
        if ([string]::IsNullOrWhiteSpace($BusId)) {
            throw (
                "Expected exactly one connected $Label matching " +
                "'$DescriptionPattern', found $($candidateDevices.Count). " +
                "An explicit bus ID is required when multiple devices match."
            )
        }
        throw "USB bus ID '$BusId' is not a connected $Label."
    }
    return $candidateDevices[0]
}


function Get-ConnectedDeviceByInstanceId {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Devices,
        [Parameter(Mandatory = $true)]
        [string]$InstanceId
    )

    $candidateDevices = @(
        $Devices | Where-Object {
            ($null -ne $_.BusId) -and ($_.InstanceId -eq $InstanceId)
        }
    )
    if ($candidateDevices.Count -ne 1) {
        throw "USB device '$InstanceId' disappeared while preparing it."
    }
    return $candidateDevices[0]
}


function Invoke-ElevatedBind {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$BusIds
    )

    foreach ($busId in $BusIds) {
        if ($busId -notmatch "^[0-9]+-[0-9]+(?:\.[0-9]+)*$") {
            throw "Invalid USB bus ID '$busId'."
        }
    }

    $powerShellPath = (Get-Process -Id $PID).Path
    $joinedBusIds = $BusIds -join ","
    $argumentList = (
        '-NoProfile -ExecutionPolicy Bypass -File "{0}" -BindBusIds "{1}"' -f
        $PSCommandPath.Replace('"', '`"'), $joinedBusIds
    )
    Write-Host "Windows UAC confirmation is required to share USB devices."
    $process = Start-Process `
        -FilePath $powerShellPath `
        -Verb RunAs `
        -ArgumentList $argumentList `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Elevated usbipd bind failed with exit code $($process.ExitCode)."
    }
}


$script:UsbipdExecutable = Get-UsbipdExecutable

if (-not [string]::IsNullOrWhiteSpace($BindBusIds)) {
    if (-not (Test-IsAdministrator)) {
        throw "The bind-only mode must run with Administrator privileges."
    }
    foreach ($busId in ($BindBusIds -split ",")) {
        if ($busId -notmatch "^[0-9]+-[0-9]+(?:\.[0-9]+)*$") {
            throw "Invalid USB bus ID '$busId'."
        }
        Invoke-Usbipd -Arguments @("bind", "--busid", $busId)
    }
    exit 0
}

$state = Get-UsbipdState
$l515 = Get-ConnectedDevice `
    -Devices $state.Devices `
    -Label "RealSense L515" `
    -DescriptionPattern $L515DescriptionPattern
$s100 = Get-ConnectedDevice `
    -Devices $state.Devices `
    -Label "S100 CH9102 serial adapter" `
    -DescriptionPattern $S100DescriptionPattern `
    -BusId $S100BusId

$targets = @(
    [PSCustomObject]@{ Label = "L515"; InstanceId = $l515.InstanceId },
    [PSCustomObject]@{ Label = "S100"; InstanceId = $s100.InstanceId }
)
$selectedDevices = @($l515, $s100)
$busIdsToBind = @(
    $selectedDevices |
        Where-Object { $null -eq $_.PersistedGuid } |
        ForEach-Object { [string]$_.BusId }
)

if ($busIdsToBind.Count -gt 0) {
    Invoke-ElevatedBind -BusIds $busIdsToBind
    $state = Get-UsbipdState
    $selectedDevices = @(
        foreach ($target in $targets) {
            Get-ConnectedDeviceByInstanceId `
                -Devices $state.Devices `
                -InstanceId $target.InstanceId
        }
    )
}

for ($index = 0; $index -lt $targets.Count; $index++) {
    $target = $targets[$index]
    $device = $selectedDevices[$index]
    if ($null -ne $device.ClientIPAddress) {
        Write-Host "$($target.Label) is already attached to a USB/IP client."
        continue
    }
    Write-Host "Attaching $($target.Label) at bus ID $($device.BusId) to WSL."
    Invoke-Usbipd -Arguments @(
        "attach", "--wsl", "--busid", [string]$device.BusId
    )
}

Write-Host "S100 and L515 USB preparation completed."
