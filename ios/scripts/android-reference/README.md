# Offline Android cache replay

These diagnostic utilities call the repository's compiled Android Java math and
frozen JSON models on retained device features. They do not decode media, contact
a device, change Android sources, or select a model. Keep inputs and generated
outputs outside the repository, for example under `E:/wslmac/artifacts`.

`CacheInference` requires a complete `native-features-v1` cache directory containing
`visual-complete.bin` (schema 2), `audio.bin` and `context.bin`, and the exact
`analyzedDurationSeconds` from its manifest. It writes `android-analysis.json`
(104 base columns, 520 contextual columns, both models' three heads and intervals)
and `identity-schema.txt` beside the copied cache. Outputs are derived and replaced
on rerun. Check `regeneratedContextDifferences=0`; a mismatch requires investigation.
This replay does not establish the source identity: independently verify the cache
key and source fingerprint before calling a comparison same-source parity.

`AudioAttribution` additionally takes the iPad's raw analysis JSON (`times`, `base`,
`contextual`, `allLabels`, `previous`, `intervals`). It evaluates both combinations
of visual/temporal and audio feature blocks, and Android JVM inference on the exact
iPad context. Its purpose is to isolate numerical differences; mixed feature blocks
are diagnostic inputs, not a proposed production pipeline.

Run from the repository root in PowerShell after the normal Android debug build
has produced Java classes. Compilation below writes only the external diagnostic
class directory; it does not run Gradle or rebuild the app.

```powershell
$replayRoot = 'E:/wslmac/artifacts/android-reference'
$replayCache = Join-Path $replayRoot 'files/native-features-v1/EXACT-CACHE-KEY'
$replayDuration = '1105.817' # Read the matching manifest, do not infer from filename.
$replayRepo = (Get-Location).Path
$replayJava = 'C:/Program Files/Android/Android Studio/jbr/bin'
$replayCompiled = Join-Path $replayRepo 'android/app/build/intermediates/javac/debug/compileDebugJavaWithJavac/classes'
$replayJSON = (Get-ChildItem 'C:/Users/developer/.gradle/caches/modules-2/files-2.1/org.json/json' -Recurse -Filter '*.jar' | Select-Object -First 1).FullName
$replayAndroid = 'C:/Users/developer/AppData/Local/Android/Sdk/platforms/android-37.0/android.jar'
$replayCP = "$replayRoot/classes;$replayCompiled;$replayJSON;$replayAndroid"
& "$replayJava/javac.exe" -cp $replayCP -d "$replayRoot/classes" ios/scripts/android-reference/CacheInference.java ios/scripts/android-reference/AudioAttribution.java
if ($LASTEXITCODE -ne 0) { throw 'Diagnostic compilation failed' }
& "$replayJava/java.exe" -cp $replayCP com.volleycut.nativeanalysis.CacheInference $replayRepo $replayCache $replayDuration
if ($LASTEXITCODE -ne 0) { throw 'Cache replay failed' }
& "$replayJava/java.exe" -cp $replayCP com.volleycut.nativeanalysis.AudioAttribution $replayRepo "$replayCache/android-analysis.json" 'E:/wslmac/artifacts/analysis-25c03a1835e4.json' "$replayRoot/audio-attribution.json"
if ($LASTEXITCODE -ne 0) { throw 'Audio attribution failed' }
```

The recovered Tds6 results and source-identity proof are in
[PARITY.md](../../PARITY.md). Mac/device builds and tests remain serial under the
main workflow; these utilities need only the local JDK and already-built classes.
