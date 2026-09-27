-keep class org.opencv.** { *; }
-dontwarn org.opencv.**

# JNI entry points resolve this class/method name in production release builds.
-keep class com.volleycut.video.NativeYuvArea { *; }
