param([string]$LabRoot = 'E:\wslmac')
$ErrorActionPreference = 'Stop'
if ((Get-PSDrive E).Free -lt 25GB) { throw 'Keep at least 25 GiB free on E:' }
$archive = Join-Path $LabRoot 'artifacts/opencv-4.12.0-ios-framework.zip'
$expected = '86b42c9f141cd9169b91be2fc380b0e556a88a95c98ccc8e7aef8349ab74cf70'
if (!(Test-Path -LiteralPath $archive)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $archive) | Out-Null
    Invoke-WebRequest -Uri 'https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-ios-framework.zip' -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() -ne $expected) {
    throw 'OpenCV release checksum mismatch'
}
scp $archive ios-build-host:opencv-4.12.0-ios-framework.zip
if ($LASTEXITCODE -ne 0) { throw 'OpenCV upload failed' }
ssh ios-build-host 'mkdir -p ~/wslmac-dependencies/opencv-4.12.0; ditto -xk ~/opencv-4.12.0-ios-framework.zip ~/wslmac-dependencies/opencv-4.12.0; test -f ~/wslmac-dependencies/opencv-4.12.0/opencv2.framework/opencv2'
if ($LASTEXITCODE -ne 0) { throw 'OpenCV extraction failed' }
