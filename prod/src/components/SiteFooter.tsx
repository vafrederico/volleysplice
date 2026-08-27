import styles from "./SiteFooter.module.css";

export function SiteFooter() {
  return (
    <footer className={styles.footer}>
      <div className={styles.copy}>
        <span>Your videos stay on this device. VolleyCut does not upload them.</span>
      </div>
      <nav className={styles.links} aria-label="Legal">
        <a href="/privacy.html">Privacy Policy</a>
        <a href="/terms.html">Terms of Use</a>
      </nav>
    </footer>
  );
}
