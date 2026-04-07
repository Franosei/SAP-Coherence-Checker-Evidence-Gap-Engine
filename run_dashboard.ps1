param(
    [int]$Port = 8080,
    [switch]$Reload = $true
)

$reloadArgs = @()
if ($Reload) {
    $reloadArgs += "--reload"
}

python -m shiny run @reloadArgs --port $Port src/dashboard/app.py
