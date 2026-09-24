plugins { id("com.android.application") }

// Research-only package: never replaces the user's production app.
android {
    namespace = "com.volleycut.neuralbenchmark"
    compileSdk = 37
    defaultConfig {
        applicationId = "com.volleycut.neuralbenchmark"
        minSdk = 34
        targetSdk = 37
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    sourceSets.getByName("main").java.srcDir("../video-common/src/main/java")
}
providers.environmentVariable("VOLLEYCUT_BENCH_BUILD_DIR").orNull?.let {
    layout.buildDirectory.set(file(it))
}
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu-api:1.4.2")
}
