-keep class org.opencv.** { *; }
-dontwarn org.opencv.**

# ONNX JNI constructs tensor/value/exception classes by their Java names. The
# 1.30 AAR's consumer rules only cover telemetry; retain the runtime API too.
-keep class ai.onnxruntime.** { *; }

# JNI entry points resolve this class/method name in production release builds.
-keep class com.volleycut.video.NativeYuvArea { *; }
