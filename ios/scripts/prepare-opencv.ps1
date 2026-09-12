param(
    [string]$LabRoot = $env:VOLLEYCUT_IOS_LAB_ROOT,
    [string]$SshHost = $env:VOLLEYCUT_IOS_SSH_HOST,
    [string]$RemoteDependencyRoot = $env:VOLLEYCUT_IOS_REMOTE_DEPENDENCY_ROOT
)
$ErrorActionPreference = 'Stop'
if (-not $LabRoot) { throw 'Set VOLLEYCUT_IOS_LAB_ROOT or pass -LabRoot' }
if (-not $SshHost) { throw 'Set VOLLEYCUT_IOS_SSH_HOST or pass -SshHost' }
if (-not $RemoteDependencyRoot) { throw 'Set VOLLEYCUT_IOS_REMOTE_DEPENDENCY_ROOT or pass -RemoteDependencyRoot' }
$labDrive = Split-Path -Qualifier ([IO.Path]::GetFullPath($LabRoot))
if ($labDrive -and (Get-PSDrive $labDrive.TrimEnd(':')).Free -lt 25GB) { throw "Keep at least 25 GiB free on $labDrive" }
$archive = Join-Path $LabRoot 'artifacts/opencv-4.12.0-ios-framework.zip'
$expected = '86b42c9f141cd9169b91be2fc380b0e556a88a95c98ccc8e7aef8349ab74cf70'
if (!(Test-Path -LiteralPath $archive)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $archive) | Out-Null
    Invoke-WebRequest -Uri 'https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-ios-framework.zip' -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() -ne $expected) {
    throw 'OpenCV release checksum mismatch'
}
scp $archive "${SshHost}:opencv-4.12.0-ios-framework.zip"
if ($LASTEXITCODE -ne 0) { throw 'OpenCV upload failed' }
ssh $SshHost "mkdir -p '$RemoteDependencyRoot/opencv-4.12.0'; ditto -xk ~/opencv-4.12.0-ios-framework.zip '$RemoteDependencyRoot/opencv-4.12.0'; test -f '$RemoteDependencyRoot/opencv-4.12.0/opencv2.framework/opencv2'"
if ($LASTEXITCODE -ne 0) { throw 'OpenCV extraction failed' }
