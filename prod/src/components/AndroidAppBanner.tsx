import { isAndroidBrowser } from "@/lib/on-device/browser-support";

import styles from "./AndroidAppBanner.module.css";

const APK_FILENAME = "VolleyCut-v0.10.4-arm64-release-signed.apk";
const APK_URL = `${import.meta.env.BASE_URL}android/${APK_FILENAME}`;

export function AndroidAppBanner() {
  if (!isAndroidBrowser()) return null;

  return (
    <aside className={styles.banner} aria-label="Android app download">
      <div className={styles.message}>
        <span className={styles.eyebrow}>NATIVE ANDROID APP</span>
        <strong>Faster analysis on Android.</strong>
        <p>
          Download VolleyCut for better speeds and a smoother editing
          experience.
        </p>
      </div>
      <a
        className={styles.download}
        href={APK_URL}
        download={APK_FILENAME}
        aria-label="Download VolleyCut v0.10.4 signed ARM64 APK"
      >
        <span>Download APK</span>
        <small>v0.10.4 · ARM64 · SIGNED</small>
      </a>
    </aside>
  );
}
