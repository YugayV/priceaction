param(
    [string]$BridgePath = ".\outputs\tradingview_ai_bridge.txt",
    [string]$ConfigPath = "product_config.example.json",
    [switch]$RunInfer
)

$ErrorActionPreference = "Stop"

function Resolve-LocalPath {
    param([string]$PathValue)

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }

    return Join-Path -Path $PSScriptRoot -ChildPath $PathValue
}

if ($RunInfer) {
    $resolvedConfigPath = Resolve-LocalPath -PathValue $ConfigPath
    if (-not (Test-Path -LiteralPath $resolvedConfigPath)) {
        throw "Config file not found: $resolvedConfigPath"
    }

    Write-Host "Running inference with config: $resolvedConfigPath"
    py (Join-Path $PSScriptRoot "run_infer.py") $resolvedConfigPath

    if ($LASTEXITCODE -ne 0) {
        throw "Inference failed with exit code $LASTEXITCODE"
    }
}

$resolvedBridgePath = Resolve-LocalPath -PathValue $BridgePath
if (-not (Test-Path -LiteralPath $resolvedBridgePath)) {
    throw "Bridge payload file not found: $resolvedBridgePath"
}

$payload = (Get-Content -LiteralPath $resolvedBridgePath -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($payload)) {
    throw "Bridge payload is empty: $resolvedBridgePath"
}

if (-not $payload.StartsWith("AIBRIDGE|")) {
    throw "Unexpected bridge payload format in: $resolvedBridgePath"
}

Set-Clipboard -Value $payload

Write-Host ""
Write-Host "TradingView AI bridge copied to clipboard."
Write-Host "Source: $resolvedBridgePath"
Write-Host ""
Write-Host $payload
