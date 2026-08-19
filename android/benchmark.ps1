[CmdletBinding()]
param(
    [string]$VideoName = "1080p60.mp4",
    [string]$DeviceSerial = "",
    [ValidateSet("All", "Video", "Audio", "Inference", "VideoAudio", "VideoInference", "AudioInference")]
    [string]$Stages = "All",
    [ValidateRange(0, 86400)]
    [double]$DurationSeconds = 0,
    [ValidateRange(1, 1000000)]
    [int]$FrameLimit = 1000,
    [ValidateRange(-1, 2147483647)]
    [int]$OperatingRate = 240,
    [ValidateSet(-1, 0, 1)]
    [int]$CodecPriority = 1,
    [ValidateSet("Use", "Bypass", "Refresh")]
    [string]$CacheMode = "Use",
    [ValidateSet("Auto", "Single", "Batched")]
    [string]$AudioDecoderMode = "Single",
    [ValidateRange(1, 100)]
    [int]$Runs = 1,
    [ValidateRange(10, 3600)]
    [int]$TimeoutSeconds = 300,
    [switch]$SummaryOnly,
    [switch]$SkipBuild,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$bundledJdk = "C:\Program Files\Android\Android Studio\jbr"
$defaultAndroidSdk = Join-Path $env:LOCALAPPDATA "Android\Sdk"
if (-not $env:JAVA_HOME -and (Test-Path -LiteralPath (Join-Path $bundledJdk "bin\java.exe"))) {
    $env:JAVA_HOME = $bundledJdk
}
if (-not $env:ANDROID_HOME -and (Test-Path -LiteralPath $defaultAndroidSdk)) {
    $env:ANDROID_HOME = $defaultAndroidSdk
}
$packageName = "com.volleycut.nativeanalysis.debug"
$activityName = "$packageName/.MainActivity"
$resultFile = "files/benchmark-result.json"
$apkPath = Join-Path $PSScriptRoot "app\build\outputs\apk\debug\app-debug.apk"
$adbTargetArguments = if ($DeviceSerial) { @("-s", $DeviceSerial) } else { @() }
$stageWireName = switch ($Stages) {
    "Video" { "video" }
    "Audio" { "audio" }
    "Inference" { "inference" }
    "VideoAudio" { "video,audio" }
    "VideoInference" { "video,inference" }
    "AudioInference" { "audio,inference" }
    default { "video,audio,inference" }
}
$durationMilliseconds = [int][math]::Round($DurationSeconds * 1000)

function Invoke-Adb {
    param([string[]]$AdbArguments)
    $output = & adb @adbTargetArguments @AdbArguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "adb $($AdbArguments -join ' ') failed:`n$($output -join "`n")"
    }
    return ($output -join "`n").Trim()
}

function Get-Median {
    param([double[]]$Values)
    $ordered = @($Values | Sort-Object)
    if ($ordered.Count % 2 -eq 1) {
        return $ordered[[int][math]::Floor($ordered.Count / 2)]
    }
    $upper = $ordered.Count / 2
    return ($ordered[$upper - 1] + $ordered[$upper]) / 2
}

function Read-BenchmarkResult {
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        $output = & adb @adbTargetArguments exec-out run-as $packageName cat $resultFile 2>$null
        $readExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($readExitCode -ne 0 -or -not $output) { return $null }
    return ($output -join "`n").Trim()
}

& adb start-server | Out-Null
$deviceLines = (& adb devices) -split "\r?\n" |
    Where-Object { $_ -match "\sdevice$" }
if ($DeviceSerial -and -not ($deviceLines | Where-Object { $_ -match "^$([regex]::Escape($DeviceSerial))\s+device$" })) {
    throw "ADB device '$DeviceSerial' is not connected."
}
if (-not $DeviceSerial -and $deviceLines.Count -ne 1) {
    throw "Expected exactly one adb device, found $($deviceLines.Count)."
}

# Keep the display awake whenever the benchmark phone is powered by USB, AC, or wireless charging.
Invoke-Adb -AdbArguments @("shell", "settings", "put", "global", "stay_on_while_plugged_in", "7") | Out-Null
Invoke-Adb -AdbArguments @("shell", "input", "keyevent", "KEYCODE_WAKEUP") | Out-Null
Invoke-Adb -AdbArguments @("shell", "wm", "dismiss-keyguard") | Out-Null

if (-not $SkipBuild) {
    Push-Location $PSScriptRoot
    try {
        & .\gradlew.bat testDebugUnitTest assembleDebug
        if ($LASTEXITCODE -ne 0) { throw "Gradle build failed." }
    } finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $apkPath)) {
    throw "APK not found at $apkPath"
}
if (-not $SkipInstall) {
    Invoke-Adb -AdbArguments @("install", "-r", $apkPath) | Out-Null
}

$mediaRows = Invoke-Adb -AdbArguments @(
    "shell", "content", "query",
    "--uri", "content://media/external/video/media",
    "--projection", "_id:_display_name"
)
$matchingRows = @(($mediaRows -split "\r?\n") | Where-Object {
    $parts = $_ -split "_display_name=", 2
    $parts.Count -eq 2 -and $parts[1] -eq $VideoName
})
if ($matchingRows.Count -eq 0) {
    throw "MediaStore has no video named '$VideoName'."
}
$selectedRow = $matchingRows[-1]
if ($selectedRow -notmatch "_id=(\d+)") {
    throw "Could not parse the MediaStore id from: $selectedRow"
}
$sourceUri = "content://media/external/video/media/$($Matches[1])"
Write-Host "Benchmark source: $VideoName ($sourceUri)"
Write-Host "Benchmark stages: $stageWireName; duration: $(if ($DurationSeconds -gt 0) { "$DurationSeconds s" } else { "full selected scope" })"
Write-Host "Audio decoder mode: $($AudioDecoderMode.ToLowerInvariant())"

$results = @()
for ($run = 1; $run -le $Runs; $run++) {
    $runId = [guid]::NewGuid().ToString("N")
    Invoke-Adb -AdbArguments @("shell", "am", "force-stop", $packageName) | Out-Null
    $launchOutput = Invoke-Adb -AdbArguments @(
        "shell", "am", "start", "-W",
        "-n", $activityName,
        "-a", "android.intent.action.VIEW",
        "-d", $sourceUri,
        "-f", "0x1",
        "--ez", "benchmark_auto_run", "true",
        "--es", "benchmark_run_id", $runId,
        "--ei", "benchmark_source_frame_limit", $FrameLimit.ToString(),
        "--ei", "benchmark_codec_operating_rate", $OperatingRate.ToString(),
        "--ei", "benchmark_codec_priority", $CodecPriority.ToString(),
        "--es", "benchmark_feature_cache_mode", $CacheMode.ToLowerInvariant(),
        "--es", "benchmark_stages", $stageWireName,
        "--es", "benchmark_audio_decoder_mode", $AudioDecoderMode.ToLowerInvariant(),
        "--ei", "benchmark_duration_milliseconds", $durationMilliseconds.ToString()
    )
    if ($launchOutput -notmatch "Status: ok") {
        throw "Activity launch did not report success:`n$launchOutput"
    }

    Write-Host "Run $run/$Runs started: $runId"
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $completed = $false
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $jsonText = Read-BenchmarkResult
        if ($jsonText) {
            try {
                $payload = $jsonText | ConvertFrom-Json
                if ($payload.benchmarkRunId -eq $runId) {
                    if ($payload.benchmarkStatus -eq "complete") {
                        $results += $payload
                        if (-not $SummaryOnly) { Write-Output $jsonText }
                        $completed = $true
                        break
                    }
                    if ($payload.benchmarkStatus -in @("failed", "cancelled")) {
                        throw "Benchmark $($payload.benchmarkStatus): $($payload.errorType): $($payload.errorMessage)"
                    }
                }
            } catch [System.ArgumentException] {
                # The app replaces the result atomically, but tolerate an incomplete read defensively.
            }
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $completed) {
        $logTail = & adb @adbTargetArguments logcat -d -t 120 -s VolleyCutBenchmark VolleyCut 2>&1
        throw "Benchmark timed out after $TimeoutSeconds seconds.`n$($logTail -join "`n")"
    }
}

$stageKeys = [ordered]@{
    video = "video_decode_and_features"
    audio = "audio_decode_and_features"
    inference = "inference"
}
$summaryParts = @()
foreach ($stageName in $stageKeys.Keys) {
    if (($stageWireName -split ",") -contains $stageName) {
        $key = $stageKeys[$stageName]
        $values = @($results | ForEach-Object { [double]$_.stageMilliseconds.$key })
        $summaryParts += ("median {0} {1:N1} ms" -f $stageName, (Get-Median -Values $values))
    }
}
$totalTimes = @($results | ForEach-Object { [double]$_.totalMilliseconds })
$summaryParts += ("median total {0:N1} ms" -f (Get-Median -Values $totalTimes))
Write-Host ("Completed {0} run(s): {1}" -f $results.Count, ($summaryParts -join ", "))
