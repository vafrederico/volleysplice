import java.security.MessageDigest
import groovy.json.JsonSlurper

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

val targetAbis = providers.gradleProperty("volleycut.abis")
    .orElse("arm64-v8a")
    .get()
    .split(',')
    .map(String::trim)
    .distinct()
check(targetAbis.isNotEmpty() && targetAbis.all { it in setOf("arm64-v8a", "x86_64") }) {
    "volleycut.abis must be a comma-separated list containing only arm64-v8a or x86_64"
}

// Resolve generated ProGuard inputs against the redirected directory too.
// getDefaultProguardFile captures this location while configuring android below.
providers.environmentVariable("VOLLEYCUT_BENCH_BUILD_DIR").orNull?.let {
    layout.buildDirectory.set(file(it).resolveSibling("pipeline-build"))
}

android {
    namespace = "com.volleycut.nativeanalysis"
    compileSdk = providers.gradleProperty("volleycut.compileSdk").orElse("37").get().toInt()
    ndkVersion = "29.0.14206865"

    externalNativeBuild {
        ndkBuild {
            path = file("src/main/jni/Android.mk")
            providers.environmentVariable("VOLLEYCUT_BENCH_BUILD_DIR").orNull?.let {
                buildStagingDirectory = file(it).resolveSibling("native-staging")
            }
        }
    }

    defaultConfig {
        applicationId = "com.volleycut.nativeanalysis"
        // Media processing uses the modern foreground-service timeout callback.
        // Android 14 is the minimum supported platform for the release build.
        minSdk = 34
        targetSdk = providers.gradleProperty("volleycut.targetSdk").orElse("37").get().toInt()
        versionCode = 30
        versionName = "0.10.16"

        buildConfigField("boolean", "BLACK_VIDEO_PREVIEW", "false")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Keep phone builds compact by default. Emulator measurements explicitly
        // opt in with -Pvolleycut.abis=x86_64.
        ndk {
            abiFilters += targetAbis
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = if (providers.gradleProperty("pipelineBenchmark").isPresent) ".pipelinebenchmark" else ".debug"
            versionNameSuffix = "-debug"
        }
        release {
            isMinifyEnabled = true
            ndk.debugSymbolLevel = "SYMBOL_TABLE"
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
        create("screenshot") {
            // Android will not install the unsigned release APK on an emulator.
            // This release-equivalent build stays non-debuggable/minified while
            // using the standard debug certificate only for repeatable captures.
            initWith(getByName("release"))
            applicationIdSuffix = ".screenshot"
            signingConfig = signingConfigs.getByName("debug")
            matchingFallbacks += listOf("release")
            buildConfigField("boolean", "BLACK_VIDEO_PREVIEW", "true")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets.getByName("main").java.srcDir("../video-common/src/main/java")
    sourceSets.getByName("main").java.srcDir("../neural-runtime/src/main/java")

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        jniLibs.useLegacyPackaging = false
    }
}

if (providers.gradleProperty("pipelineBenchmark").isPresent) {
    android.sourceSets.getByName("debug").java.srcDir("src/pipelineBenchmark/java")
    android.sourceSets.getByName("debug").manifest.srcFile("src/pipelineBenchmark/AndroidManifest.xml")
    android.sourceSets.getByName("debug").java.srcDir("../neuralbenchmark/src/main/java")
    dependencies {
        add("debugImplementation", "com.google.ai.edge.litert:litert:1.4.2")
        add("debugImplementation", "com.google.ai.edge.litert:litert-gpu:1.4.2")
        add("debugImplementation", "com.google.ai.edge.litert:litert-gpu-api:1.4.2")
    }
}

val generatedNeuralAssets = layout.buildDirectory.dir("generated/neural-assets")
android.sourceSets.getByName("main").assets.srcDir(generatedNeuralAssets.get().asFile)
val prepareNeuralAssets by tasks.registering {
    val manifest = rootProject.file("../models/distilled-large/android-manifest.json")
    val bundleRoot = providers.environmentVariable("VOLLEYCUT_NEURAL_ASSETS_DIR")
    inputs.file(manifest)
    inputs.dir(rootProject.file("../models/distilled-large/licenses"))
    inputs.property("bundleRoot", bundleRoot.orElse(""))
    bundleRoot.orNull?.let { inputs.dir(file(it).resolve("android/rally-models")) }
    outputs.dir(generatedNeuralAssets)
    doLast {
        val source = bundleRoot.orNull?.let { file(it).resolve("android/rally-models") }
            ?: error("Set VOLLEYCUT_NEURAL_ASSETS_DIR to the prepared, verified neural bundle root")
        val expected = JsonSlurper().parse(manifest) as Map<*, *>
        val actual = JsonSlurper().parse(source.resolve("manifest.json")) as Map<*, *>
        check(expected == actual) { "Neural bundle manifest does not match the pinned production manifest" }
        val target = generatedNeuralAssets.get().asFile.resolve("rally-models")
        target.mkdirs()
        val variants = expected["variants"] as Map<*, *>
        for (rawVariant in variants.values) {
            val variant = rawVariant as Map<*, *>
            val directory = variant["directory"] as String
            check(directory.matches(Regex("[a-z0-9-]+"))) { "Invalid variant directory" }
            for (rawFile in (variant["files"] as Map<*, *>).values) {
                val metadata = rawFile as Map<*, *>
                val name = metadata["name"] as String
                check(name.matches(Regex("[a-z0-9.-]+"))) { "Invalid model filename" }
                val input = source.resolve(directory).resolve(name)
                val digest = MessageDigest.getInstance("SHA-256").digest(input.readBytes())
                    .joinToString("") { "%02x".format(it.toInt() and 0xff) }
                check(input.length() == (metadata["sizeBytes"] as Number).toLong() && digest == metadata["sha256"]) {
                    "Neural asset integrity mismatch: $directory/$name"
                }
                val output = target.resolve(directory).resolve(name)
                output.parentFile.mkdirs()
                input.copyTo(output, overwrite = true)
            }
        }
        manifest.copyTo(target.resolve("manifest.json"), overwrite = true)
        rootProject.file("../models/distilled-large/licenses").copyRecursively(
            target.resolve("licenses"), overwrite = true,
        )
    }
}

val verifySuppressionAsset by tasks.registering {
    val asset = layout.projectDirectory.file(
        "src/main/assets/suppression-overlap-exclusion-retrained.json",
    )
    inputs.file(asset)
    doLast {
        val expected = "02274d0f17b89cd54ea24da7d1665a6475e48dc0f554d06092e9ef892332d4f1"
        val actual = MessageDigest.getInstance("SHA-256")
            .digest(asset.asFile.readBytes())
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        check(actual == expected) {
            "Frozen suppression asset hash mismatch: expected $expected, got $actual"
        }
    }
}

val verifyServingSideAsset by tasks.registering {
    val asset = layout.projectDirectory.file(
        "src/main/assets/serving-side-85bc3325fbd4.json",
    )
    inputs.file(asset)
    doLast {
        val expected = "14f18bf0b0f326ccd7ef4b3d614a96a53dd9675df61813fd375677489d0e5a7c"
        // Git may materialize CRLF in an existing Windows checkout. The runtime
        // identity is the canonical LF JSON payload used to build the model.
        val canonical = asset.asFile.readText(Charsets.UTF_8)
            .replace("\r\n", "\n")
            .toByteArray(Charsets.UTF_8)
        val actual = MessageDigest.getInstance("SHA-256")
            .digest(canonical)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        check(actual == expected) {
            "Frozen serving-side asset hash mismatch: expected $expected, got $actual"
        }
    }
}

val verifySideSwitchAsset by tasks.registering {
    val asset = layout.projectDirectory.file(
        "src/main/assets/side-switch-c2570481c30d.json",
    )
    inputs.file(asset)
    doLast {
        val expected = "ab4197545fb916a37ee6ac1d69e74ddfc0123c09039cdfa88c4ef378dd3e27fc"
        val canonical = asset.asFile.readText(Charsets.UTF_8)
            .replace("\r\n", "\n")
            .toByteArray(Charsets.UTF_8)
        val actual = MessageDigest.getInstance("SHA-256")
            .digest(canonical)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        check(actual == expected) {
            "Frozen team-switch asset hash mismatch: expected $expected, got $actual"
        }
    }
}

tasks.named("preBuild").configure {
    dependsOn(verifySuppressionAsset, verifyServingSideAsset, verifySideSwitchAsset, prepareNeuralAssets)
}

dependencies {
    implementation("org.opencv:opencv:4.12.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")

    val composeBom = platform("androidx.compose:compose-bom:2026.06.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    val media3Version = "1.10.1"
    implementation("androidx.media3:media3-common:$media3Version")
    implementation("androidx.media3:media3-exoplayer:$media3Version")
    implementation("androidx.media3:media3-ui-compose:$media3Version")
    implementation("androidx.media3:media3-effect:$media3Version")
    implementation("androidx.media3:media3-transformer:$media3Version")
    implementation("androidx.media3:media3-muxer:$media3Version")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
    androidTestImplementation("androidx.test.uiautomator:uiautomator:2.4.0")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
}
