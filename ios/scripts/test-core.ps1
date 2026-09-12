param([string]$LabRoot = 'E:\wslmac')
$ErrorActionPreference = 'Stop'
python "$PSScriptRoot/prepare-source.py" --output "$LabRoot/artifacts/volleysplice-source.tar.gz"
if ($LASTEXITCODE -ne 0) { throw 'Source preparation failed' }
scp "$LabRoot/artifacts/volleysplice-source.tar.gz" ios-build-host:volleysplice-source.tar.gz
if ($LASTEXITCODE -ne 0) { throw 'Upload failed' }
ssh ios-build-host 'mkdir -p ~/wslmac-projects/volleysplice; tar -xzf ~/volleysplice-source.tar.gz -C ~/wslmac-projects/volleysplice; cd ~/wslmac-projects/volleysplice; swift test -j 4'
if ($LASTEXITCODE -ne 0) { throw 'Swift tests failed' }
