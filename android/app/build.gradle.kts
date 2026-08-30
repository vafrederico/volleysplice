import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.volleycut.nativeanalysis"
    compileSdk = providers.gradleProperty("volleycut.compileSdk").orElse("37").get().toInt()

    defaultConfig {
        applicationId = "com.volleycut.nativeanalysis"
        // Media processing uses the modern foreground-service timeout callback.
        // Android 14 is the minimum supported platform for the release build.
        minSdk = 34
        targetSdk = providers.gradleProperty("volleycut.targetSdk").orElse("37").get().toInt()
        versionCode = 25
        versionName = "0.10.11"

        buildConfigField("boolean", "BLACK_VIDEO_PREVIEW", "false")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // The benchmark target is a Pixel 10 Pro. Keeping only arm64 avoids a
        // very large APK containing OpenCV binaries for emulator architectures.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
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

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        jniLibs.useLegacyPackaging = false
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
    dependsOn(verifySuppressionAsset, verifyServingSideAsset, verifySideSwitchAsset)
}

dependencies {
    implementation("org.opencv:opencv:4.12.0")

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
