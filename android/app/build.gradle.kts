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
        minSdk = 29
        targetSdk = providers.gradleProperty("volleycut.targetSdk").orElse("37").get().toInt()
        versionCode = 14
        versionName = "0.10.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // The benchmark target is a Pixel 10 Pro. Keeping only arm64 avoids a
        // very large APK containing OpenCV binaries for emulator architectures.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
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

tasks.named("preBuild").configure { dependsOn(verifySuppressionAsset) }

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

    val media3Version = "1.10.1"
    implementation("androidx.media3:media3-common:$media3Version")
    implementation("androidx.media3:media3-exoplayer:$media3Version")
    implementation("androidx.media3:media3-ui-compose:$media3Version")
    implementation("androidx.media3:media3-transformer:$media3Version")
    implementation("androidx.media3:media3-muxer:$media3Version")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
}
