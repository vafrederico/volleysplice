[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$AabPath,

    [string]$KeystorePath = "$env:USERPROFILE\.android-keystores\volleycut-release.jks",
    [string]$KeyAlias = "volleycut-release",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

$aab = (Resolve-Path -LiteralPath $AabPath).Path
$keystore = (Resolve-Path -LiteralPath $KeystorePath).Path

if ([IO.Path]::GetExtension($aab) -ne ".aab") {
    throw "The input file must be an Android App Bundle (.aab)."
}

$bundledJdk = "C:\Program Files\Android\Android Studio\jbr"
if (-not $env:JAVA_HOME -and (Test-Path -LiteralPath (Join-Path $bundledJdk "bin\java.exe"))) {
    $env:JAVA_HOME = $bundledJdk
}
if (-not $env:JAVA_HOME -or -not (Test-Path -LiteralPath (Join-Path $env:JAVA_HOME "bin\java.exe"))) {
    throw "Java was not found. Set JAVA_HOME or install Android Studio."
}

$jarsigner = Join-Path $env:JAVA_HOME "bin\jarsigner.exe"
if (-not (Test-Path -LiteralPath $jarsigner)) {
    throw "jarsigner was not found under '$env:JAVA_HOME'."
}

if (-not $OutputPath) {
    $directory = Split-Path -Parent $aab
    $name = [IO.Path]::GetFileNameWithoutExtension($aab) -replace "-unsigned$", ""
    $OutputPath = Join-Path $directory "$name-signed.aab"
}
$OutputPath = [IO.Path]::GetFullPath($OutputPath)

if ($OutputPath -eq $aab) {
    throw "OutputPath must be different from the unsigned AAB path."
}

# jarsigner prompts for the keystore password locally. Do not add passwords here.
& $jarsigner `
    -verbose `
    -sigalg SHA256withRSA `
    -digestalg SHA-256 `
    -keystore $keystore `
    -signedjar $OutputPath `
    $aab `
    $KeyAlias
if ($LASTEXITCODE -ne 0) {
    throw "AAB signing failed."
}

& $jarsigner -verify -verbose -certs $OutputPath
if ($LASTEXITCODE -ne 0) {
    throw "The signed AAB failed verification."
}

Write-Host "Signed AAB: $OutputPath"
