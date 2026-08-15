plugins {
    id("com.android.application")
}

android {
    namespace = "com.volleycut.nativeanalysis"
    compileSdk = providers.gradleProperty("volleycut.compileSdk").orElse("36").get().toInt()

    defaultConfig {
        applicationId = "com.volleycut.nativeanalysis"
        minSdk = 29
        targetSdk = providers.gradleProperty("volleycut.targetSdk").orElse("36").get().toInt()
        versionCode = 11
        versionName = "0.7.0"

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

    packaging {
        jniLibs.useLegacyPackaging = false
    }
}

dependencies {
    implementation("org.opencv:opencv:4.12.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
}
