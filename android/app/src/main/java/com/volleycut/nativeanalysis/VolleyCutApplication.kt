package com.volleycut.nativeanalysis

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader

class VolleyCutApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "Could not initialize the native OpenCV runtime")
        }
    }

    companion object {
        private const val TAG = "VolleyCut"
    }
}
