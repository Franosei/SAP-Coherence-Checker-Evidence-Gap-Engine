param(
    [int]$Port = 8030,
    [switch]$Reload = $false
)

$dashboardPattern = '*src/dashboard/app.py*'
$existing = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like $dashboardPattern }

if ($existing) {
    $ids = $existing.ProcessId | Sort-Object -Unique
    Write-Host "Stopping existing dashboard process(es): $($ids -join ', ')"
    Stop-Process -Id $ids -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

$reloadArgs = @()
if ($Reload) {
    $reloadArgs += "--reload"
}

Write-Host "Starting dashboard on http://127.0.0.1:$Port"
python -m shiny run @reloadArgs --port $Port src/dashboard/app.py
