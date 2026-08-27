package com.volleycut.nativeanalysis

import android.content.Context
import android.content.Intent
import android.net.Uri

internal object AppRating {
    private const val STORE_PACKAGE_ID = "com.volleycut.nativeanalysis"
    private const val PREFERENCES = "app_rating"
    private const val COMPLETED_EXPORTS = "completed_exports"
    private const val LAST_COMPLETED_JOB_ID = "last_completed_job_id"
    private const val NEXT_PROMPT_EXPORT = "next_prompt_export"
    private const val PROMPT_PENDING = "prompt_pending"
    private const val RATING_OPENED = "rating_opened"
    private const val EXPORTS_BETWEEN_PROMPTS = 3

    @Synchronized
    fun recordSuccessfulExport(context: Context, jobId: String) {
        val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        if (preferences.getBoolean(RATING_OPENED, false) ||
            preferences.getString(LAST_COMPLETED_JOB_ID, null) == jobId
        ) return

        val completedExports = preferences.getInt(COMPLETED_EXPORTS, 0) + 1
        val nextPromptExport = preferences.getInt(NEXT_PROMPT_EXPORT, 1)
        preferences.edit()
            .putInt(COMPLETED_EXPORTS, completedExports)
            .putString(LAST_COMPLETED_JOB_ID, jobId)
            .putBoolean(PROMPT_PENDING, completedExports >= nextPromptExport)
            .apply()
    }

    fun hasPendingPrompt(context: Context): Boolean {
        val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        return !preferences.getBoolean(RATING_OPENED, false) &&
            preferences.getBoolean(PROMPT_PENDING, false)
    }

    fun deferPrompt(context: Context) {
        val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val completedExports = preferences.getInt(COMPLETED_EXPORTS, 0)
        preferences.edit()
            .putInt(NEXT_PROMPT_EXPORT, completedExports + EXPORTS_BETWEEN_PROMPTS)
            .putBoolean(PROMPT_PENDING, false)
            .apply()
    }

    fun openPlayStore(context: Context): Boolean {
        val marketIntent = Intent(
            Intent.ACTION_VIEW,
            Uri.parse("market://details?id=$STORE_PACKAGE_ID"),
        ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        val webIntent = Intent(
            Intent.ACTION_VIEW,
            Uri.parse("https://play.google.com/store/apps/details?id=$STORE_PACKAGE_ID"),
        ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        val opened = runCatching {
            context.startActivity(marketIntent)
            true
        }.getOrElse {
            runCatching {
                context.startActivity(webIntent)
                true
            }.getOrDefault(false)
        }
        if (opened) {
            context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
                .edit()
                .putBoolean(RATING_OPENED, true)
                .putBoolean(PROMPT_PENDING, false)
                .apply()
        }
        return opened
    }
}
