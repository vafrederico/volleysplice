param(
    [string]$LabRoot = $env:VOLLEYCUT_IOS_LAB_ROOT,
    [string]$SshHost = $env:VOLLEYCUT_IOS_SSH_HOST,
    [string]$RemoteProjectRoot = $env:VOLLEYCUT_IOS_REMOTE_PROJECT_ROOT,
    [string]$RemoteUser = $env:VOLLEYCUT_IOS_REMOTE_USER,
    [string]$RemoteUid = $env:VOLLEYCUT_IOS_REMOTE_UID,
    [switch]$Install,
    [switch]$Unsigned
)
$ErrorActionPreference = 'Stop'
if ($Unsigned -and $Install) { throw 'Unsigned builds cannot be installed' }
if (-not $LabRoot) { throw 'Set VOLLEYCUT_IOS_LAB_ROOT or pass -LabRoot' }
if (-not $SshHost) { throw 'Set VOLLEYCUT_IOS_SSH_HOST or pass -SshHost' }
if (-not $RemoteProjectRoot) { throw 'Set VOLLEYCUT_IOS_REMOTE_PROJECT_ROOT or pass -RemoteProjectRoot' }
if ($SshHost -notmatch '^[A-Za-z0-9._-]+$') { throw 'Unsafe SSH host alias' }
if ($RemoteProjectRoot.Contains("'")) { throw 'Remote project root cannot contain a single quote' }
$deviceUdid = $env:VOLLEYCUT_IOS_DEVICE_UDID
$developmentTeam = $env:VOLLEYCUT_IOS_DEVELOPMENT_TEAM
$openCvFrameworkRoot = $env:VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT
if (-not $openCvFrameworkRoot) { throw 'Set VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT' }
if (-not $Unsigned) {
    if (-not $deviceUdid) { throw 'Set VOLLEYCUT_IOS_DEVICE_UDID for a signed device build' }
    if (-not $developmentTeam) { throw 'Set VOLLEYCUT_IOS_DEVELOPMENT_TEAM for a signed device build' }
    if (-not $RemoteUser) { throw 'Set VOLLEYCUT_IOS_REMOTE_USER for a signed device build' }
    if ($RemoteUid -notmatch '^\d+$') { throw 'Set VOLLEYCUT_IOS_REMOTE_UID to the remote console user ID' }
    if ($RemoteUser -notmatch '^[A-Za-z0-9._-]+$') { throw 'Unsafe remote user' }
    if ($deviceUdid -notmatch '^[A-Fa-f0-9-]+$') { throw 'Unsafe iOS device UDID' }
    if ($developmentTeam -notmatch '^[A-Z0-9]+$') { throw 'Unsafe Apple development team ID' }
}
python "$PSScriptRoot/generate-project.py"
if ($LASTEXITCODE -ne 0) { throw 'Project generation failed' }
python "$PSScriptRoot/prepare-source.py" --output "$LabRoot/artifacts/volleysplice-source.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'Source preparation failed' }
scp "$LabRoot/artifacts/volleysplice-source.tar.gz" "${SshHost}:volleysplice-source.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed' }
ssh $SshHost "mkdir -p '$RemoteProjectRoot'; tar -xzf ~/volleysplice-source.tar.gz -C '$RemoteProjectRoot'"
if ($LASTEXITCODE -ne 0) { throw 'Unpack failed' }
if ($Unsigned) {
    ssh $SshHost "VOLLEYCUT_IOS_REMOTE_PROJECT_ROOT='$RemoteProjectRoot' VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT='$openCvFrameworkRoot' bash '$RemoteProjectRoot/scripts/build-device.sh' --unsigned"
    if ($LASTEXITCODE -ne 0) { throw 'Unsigned device build failed' }
    return
}
$remoteBuild = "launchctl asuser $RemoteUid sudo -u '$RemoteUser' env VOLLEYCUT_IOS_REMOTE_PROJECT_ROOT='$RemoteProjectRoot' VOLLEYCUT_IOS_DEVICE_UDID='$deviceUdid' VOLLEYCUT_IOS_DEVELOPMENT_TEAM='$developmentTeam' VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT='$openCvFrameworkRoot' /bin/bash '$RemoteProjectRoot/scripts/build-device.sh'"
python "$LabRoot/scripts/mac-remote.py" --sudo --command $remoteBuild
if ($LASTEXITCODE -ne 0) { throw 'Device build/sign failed' }
scp "${SshHost}:$RemoteProjectRoot/artifacts/VolleySplice.ipa" "$LabRoot/artifacts/VolleySplice.ipa"
if ($LASTEXITCODE -ne 0) { throw 'IPA download failed' }
$remoteHash = (ssh $SshHost "shasum -a 256 '$RemoteProjectRoot/artifacts/VolleySplice.ipa'").Split(' ')[0]
if ($LASTEXITCODE -ne 0) { throw 'Remote hash failed' }
if ((Get-FileHash "$LabRoot/artifacts/VolleySplice.ipa" -Algorithm SHA256).Hash.ToLower() -ne $remoteHash) { throw 'IPA hash mismatch' }
if ($Install) {
    python "$LabRoot/scripts/wda-client.py" park
    if ($LASTEXITCODE -ne 0) { throw 'Parking failed' }
    & "$LabRoot/scripts/install-ipa.ps1" -IpaPath "$LabRoot/artifacts/VolleySplice.ipa"
    if ($LASTEXITCODE -ne 0) { throw 'Installation failed' }
}
