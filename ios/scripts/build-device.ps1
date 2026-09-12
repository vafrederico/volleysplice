param([string]$LabRoot = 'E:\wslmac', [switch]$Install, [switch]$Unsigned)
$ErrorActionPreference = 'Stop'
if ($Unsigned -and $Install) { throw 'Unsigned builds cannot be installed' }
python "$PSScriptRoot/generate-project.py"
if ($LASTEXITCODE -ne 0) { throw 'Project generation failed' }
python "$PSScriptRoot/prepare-source.py" --output "$LabRoot/artifacts/volleysplice-source.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'Source preparation failed' }
scp "$LabRoot/artifacts/volleysplice-source.tar.gz" ios-build-host:volleysplice-source.tar.gz
if ($LASTEXITCODE -ne 0) { throw 'Upload failed' }
ssh ios-build-host 'mkdir -p ~/wslmac-projects/volleysplice; tar -xzf ~/volleysplice-source.tar.gz -C ~/wslmac-projects/volleysplice'
if ($LASTEXITCODE -ne 0) { throw 'Unpack failed' }
if ($Unsigned) {
    ssh ios-build-host 'bash ~/wslmac-projects/volleysplice/scripts/build-device.sh --unsigned'
    if ($LASTEXITCODE -ne 0) { throw 'Unsigned device build failed' }
    return
}
python "$LabRoot/scripts/mac-remote.py" --sudo --command 'launchctl asuser 501 sudo -u vafrederico /bin/bash /Users/developer/wslmac-projects/volleysplice/scripts/build-device.sh'
if ($LASTEXITCODE -ne 0) { throw 'Device build/sign failed' }
scp ios-build-host:wslmac-projects/volleysplice/artifacts/VolleySplice.ipa "$LabRoot/artifacts/VolleySplice.ipa"
if ($LASTEXITCODE -ne 0) { throw 'IPA download failed' }
$remoteHash = (ssh ios-build-host 'shasum -a 256 ~/wslmac-projects/volleysplice/artifacts/VolleySplice.ipa').Split(' ')[0]
if ($LASTEXITCODE -ne 0) { throw 'Remote hash failed' }
if ((Get-FileHash "$LabRoot/artifacts/VolleySplice.ipa" -Algorithm SHA256).Hash.ToLower() -ne $remoteHash) { throw 'IPA hash mismatch' }
if ($Install) {
    python "$LabRoot/scripts/wda-client.py" park
    if ($LASTEXITCODE -ne 0) { throw 'Parking failed' }
    & "$LabRoot/scripts/install-ipa.ps1" -IpaPath "$LabRoot/artifacts/VolleySplice.ipa"
    if ($LASTEXITCODE -ne 0) { throw 'Installation failed' }
}
