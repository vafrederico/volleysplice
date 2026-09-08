import { GOOGLE_PLAY_URL } from "@/lib/android-app";
import { isAndroidBrowser } from "@/lib/on-device/browser-support";

import styles from "./AndroidAppBanner.module.css";

export function AndroidAppBanner() {
  if (!isAndroidBrowser()) return null;

  return (
    <aside className={styles.banner} aria-label="Android app download">
      <div className={styles.message}>
        <span className={styles.eyebrow}>NATIVE ANDROID APP</span>
        <strong>Faster analysis on Android.</strong>
        <p>
          Get VolleyCut on Google Play for better speeds and a smoother editing
          experience.
        </p>
      </div>
      <a
        className={styles.storeLink}
        href={GOOGLE_PLAY_URL}
        aria-label="Get VolleyCut on Google Play"
      >
        <img
          src={`${import.meta.env.BASE_URL}google-play-badge.png`}
          alt="Get it on Google Play"
          width={180}
          height={70}
        />
      </a>
    </aside>
  );
}
