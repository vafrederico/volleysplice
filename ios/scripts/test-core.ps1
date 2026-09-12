param(
    [string]$LabRoot = $env:VOLLEYCUT_IOS_LAB_ROOT,
    [string]$SshHost = $env:VOLLEYCUT_IOS_SSH_HOST,
    [string]$RemoteProjectRoot = $env:VOLLEYCUT_IOS_REMOTE_PROJECT_ROOT
)
$ErrorActionPreference = 'Stop'
if (-not $LabRoot) { throw 'Set VOLLEYCUT_IOS_LAB_ROOT or pass -LabRoot' }
if (-not $SshHost) { throw 'Set VOLLEYCUT_IOS_SSH_HOST or pass -SshHost' }
if (-not $RemoteProjectRoot) { throw 'Set VOLLEYCUT_IOS_REMOTE_PROJECT_ROOT or pass -RemoteProjectRoot' }
python "$PSScriptRoot/prepare-source.py" --output "$LabRoot/artifacts/volleysplice-source.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'Source preparation failed' }
scp "$LabRoot/artifacts/volleysplice-source.tar.gz" "${SshHost}:volleysplice-source.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed' }
ssh $SshHost "mkdir -p '$RemoteProjectRoot'; tar -xzf ~/volleysplice-source.tar.gz -C '$RemoteProjectRoot'; cd '$RemoteProjectRoot'; swift test -j 4"
if ($LASTEXITCODE -ne 0) { throw 'Swift tests failed' }
