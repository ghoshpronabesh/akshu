$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Run: py -3.11 -m venv .venv"
}

try {
    $PythonVersion = & $VenvPython --version 2>&1
}
catch {
    Write-Error "Virtual environment Python is not runnable. Recreate it with: py -3.11 -m venv .venv"
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "Virtual environment Python is not runnable ($PythonVersion). Recreate it with: py -3.11 -m venv .venv"
}

Set-Location $ProjectRoot

try {
    $LocalIp = Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.IPAddress -notlike "127.*" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.InterfaceOperationalStatus -eq "Up"
        } |
        Sort-Object InterfaceMetric |
        Select-Object -First 1 -ExpandProperty IPAddress
}
catch {
    $LocalIp = $null
}

if (-not $LocalIp) {
    $LocalIp = [System.Net.Dns]::GetHostAddresses($env:COMPUTERNAME) |
        Where-Object { $_.AddressFamily -eq "InterNetwork" -and $_.IPAddressToString -notlike "127.*" } |
        Select-Object -First 1 -ExpandProperty IPAddressToString
}

if (-not $LocalIp) {
    $LocalIp = "<your-laptop-ip>"
}

Write-Host ""
Write-Host "Starting Akshu Streamlit dashboard on the LAN..."
Write-Host "Open on this laptop:  http://localhost:8501"
Write-Host "Open from same Wi-Fi: http://${LocalIp}:8501"
Write-Host ""

& $VenvPython -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
