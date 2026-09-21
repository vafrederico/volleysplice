import { APP_STORE_URL } from "@/lib/ios-app";
import { GOOGLE_PLAY_URL } from "@/lib/android-app";
import { isAndroidBrowser, isIosBrowser } from "@/lib/on-device/browser-support";

import styles from "./MobileAppBanner.module.css";

export function MobileAppBanner() {
  const ios = isIosBrowser();
  if (!ios && !isAndroidBrowser()) return null;
  const store = ios ? "App Store" : "Google Play";
  const platform = ios ? "iOS" : "Android";

  return (
    <aside className={styles.banner} aria-label={`${platform} app download`}>
      <div className={styles.message}>
        <span className={styles.eyebrow}>NATIVE {platform.toUpperCase()} APP</span>
        <strong>{ios ? "VolleySplice is on the App Store." : "Faster analysis on Android."}</strong>
        <p>
          {ios
            ? "Get the app to analyze and edit your volleyball videos on iPhone and iPad."
            : "Get VolleySplice on Google Play for better speeds and a smoother editing experience."}
        </p>
      </div>
      <a
        className={styles.storeLink}
        href={ios ? APP_STORE_URL : GOOGLE_PLAY_URL}
        aria-label={`Get VolleySplice on the ${store}`}
      >
        <img
          src={`${import.meta.env.BASE_URL}${ios ? "app-store-badge.svg" : "google-play-badge.png"}`}
          alt={ios ? "Download on the App Store" : "Get it on Google Play"}
          width={180}
          height={ios ? 60 : 70}
        />
      </a>
    </aside>
  );
}
