plugins {
    id("com.android.application") version "9.1.1" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.10" apply false
}
providers.environmentVariable("VOLLEYCUT_BENCH_BUILD_DIR").orNull?.let {
    layout.buildDirectory.set(file(it).resolveSibling("root-build"))
}
