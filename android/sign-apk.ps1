[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$ApkPath,

    [string]$KeystorePath = "$env:USERPROFILE\.android-keystores\volleycut-release.jks",
    [string]$KeyAlias = "volleycut-release",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

$apk = (Resolve-Path -LiteralPath $ApkPath).Path
$keystore = (Resolve-Path -LiteralPath $KeystorePath).Path

$bundledJdk = "C:\Program Files\Android\Android Studio\jbr"
if (-not $env:JAVA_HOME -and (Test-Path -LiteralPath (Join-Path $bundledJdk "bin\java.exe"))) {
    $env:JAVA_HOME = $bundledJdk
}
if (-not $env:JAVA_HOME -or -not (Test-Path -LiteralPath (Join-Path $env:JAVA_HOME "bin\java.exe"))) {
    throw "Java was not found. Set JAVA_HOME or install Android Studio."
}

$androidSdk = if ($env:ANDROID_HOME) {
    $env:ANDROID_HOME
} elseif ($env:ANDROID_SDK_ROOT) {
    $env:ANDROID_SDK_ROOT
} else {
    Join-Path $env:LOCALAPPDATA "Android\Sdk"
}

$buildToolsRoot = Join-Path $androidSdk "build-tools"
$buildTools = Get-ChildItem -LiteralPath $buildToolsRoot -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "apksigner.bat") } |
    Sort-Object { [version]$_.Name } -Descending |
    Select-Object -First 1
if (-not $buildTools) {
    throw "apksigner was not found under '$buildToolsRoot'."
}

$apksigner = Join-Path $buildTools.FullName "apksigner.bat"
$zipalign = Join-Path $buildTools.FullName "zipalign.exe"

if (-not $OutputPath) {
    $directory = Split-Path -Parent $apk
    $name = [IO.Path]::GetFileNameWithoutExtension($apk) -replace "-unsigned$", ""
    $OutputPath = Join-Path $directory "$name-signed.apk"
} else {
    $OutputPath = [IO.Path]::GetFullPath($OutputPath)
}

& $zipalign -c -P 16 4 $apk
if ($LASTEXITCODE -ne 0) {
    throw "The APK is not zip-aligned. Align it before signing."
}

& $apksigner sign `
    --ks $keystore `
    --ks-key-alias $KeyAlias `
    --v4-signing-enabled false `
    --out $OutputPath `
    $apk
if ($LASTEXITCODE -ne 0) {
    throw "APK signing failed."
}

& $apksigner verify --verbose --print-certs $OutputPath
if ($LASTEXITCODE -ne 0) {
    throw "The signed APK failed verification."
}

Write-Host "Signed APK: $OutputPath"
