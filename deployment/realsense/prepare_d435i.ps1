[CmdletBinding()]
param(
    [string]$ForceBindBusId = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$DescriptionPattern = "(?i)(RealSense.*D?435I|D?435I.*RealSense)"

# usbipd emits UTF-8 JSON. Windows PowerShell 5.1 otherwise decodes native
# output with the legacy console code page, which can corrupt Chinese device
# descriptions and make the complete JSON document invalid.
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom


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
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
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
    return (($jsonLines -join [Environment]::NewLine) | ConvertFrom-Json)
}


function Get-D435i {
    param([Parameter(Mandatory = $true)][object[]]$Devices)
    $matches = @(
        $Devices | Where-Object {
            ($null -ne $_.BusId) -and
            ([string]$_.Description -match $DescriptionPattern)
        }
    )
    if ($matches.Count -ne 1) {
        Write-Host "Connected USB devices reported by usbipd:"
        foreach ($device in $Devices) {
            if ($null -ne $device.BusId) {
                Write-Host "  $($device.BusId)  $($device.Description)"
            }
        }
        throw "Expected exactly one connected RealSense D435i, found $($matches.Count)."
    }
    return $matches[0]
}


function Get-UsbipdDeviceByBusId {
    param(
        [Parameter(Mandatory = $true)][object[]]$Devices,
        [Parameter(Mandatory = $true)][string]$BusId
    )
    $matches = @(
        $Devices | Where-Object {
            ($null -ne $_.BusId) -and ([string]$_.BusId -eq $BusId)
        }
    )
    if ($matches.Count -ne 1) {
        throw "Expected exactly one USB device at bus ID $BusId, found $($matches.Count)."
    }
    return $matches[0]
}


function Invoke-ElevatedForceBind {
    param([Parameter(Mandatory = $true)][string]$BusId)
    if ($BusId -notmatch "^[0-9]+-[0-9]+(?:\.[0-9]+)*$") {
        throw "Invalid USB bus ID '$BusId'."
    }
    $powerShellPath = (Get-Process -Id $PID).Path
    $argumentList = (
        '-NoProfile -ExecutionPolicy Bypass -File "{0}" -ForceBindBusId "{1}"' -f
        $PSCommandPath.Replace('"', '""'), $BusId
    )
    Write-Host "Windows UAC confirmation is required to force-bind D435i for WSL."
    $process = Start-Process -FilePath $powerShellPath -Verb RunAs -ArgumentList $argumentList -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Elevated usbipd force-bind failed with exit code $($process.ExitCode)."
    }
}


$script:UsbipdExecutable = Get-UsbipdExecutable

if (-not [string]::IsNullOrWhiteSpace($ForceBindBusId)) {
    if (-not (Test-IsAdministrator)) {
        throw "The force-bind mode must run with Administrator privileges."
    }
    $state = Get-UsbipdState
    $device = Get-UsbipdDeviceByBusId -Devices $state.Devices -BusId $ForceBindBusId
    if ($null -ne $device.ClientIPAddress) {
        Invoke-Usbipd -Arguments @("detach", "--busid", $ForceBindBusId)
        Start-Sleep -Milliseconds 500
    }
    if ($null -ne $device.PersistedGuid) {
        Invoke-Usbipd -Arguments @("unbind", "--busid", $ForceBindBusId)
        Start-Sleep -Milliseconds 500
    }
    Invoke-Usbipd -Arguments @("bind", "--force", "--busid", $ForceBindBusId)
    exit 0
}

$state = Get-UsbipdState
$d435i = Get-D435i -Devices $state.Devices
if (-not [bool]$d435i.IsForced) {
    if ($null -ne $d435i.ClientIPAddress) {
        Write-Host "Detaching D435i before replacing the normal share with a force binding."
        Invoke-Usbipd -Arguments @("detach", "--busid", [string]$d435i.BusId)
        Start-Sleep -Milliseconds 500
    }
    Invoke-ElevatedForceBind -BusId ([string]$d435i.BusId)
    $state = Get-UsbipdState
    $d435i = Get-D435i -Devices $state.Devices
    if (-not [bool]$d435i.IsForced) {
        throw "D435i force binding was not recorded by usbipd."
    }
}
if ($null -ne $d435i.ClientIPAddress) {
    Write-Host "D435i is already attached to a USB/IP client."
    exit 0
}

Write-Host "Attaching D435i at bus ID $($d435i.BusId) to WSL."
Invoke-Usbipd -Arguments @(
    "attach", "--wsl", "--busid", [string]$d435i.BusId
)
Write-Host "D435i USB preparation completed."
