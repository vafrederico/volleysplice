import styles from "./SiteFooter.module.css";

export function SiteFooter() {
  return (
    <footer className={styles.footer}>
      <div className={styles.copy}>
        <span>
          Project metadata, generated features, predictions, and edit drafts
          stay in this browser.
        </span>
        <span>
          Local video bytes are never uploaded or copied into project storage.
        </span>
      </div>
      <nav className={styles.links} aria-label="Legal">
        <a href="/privacy.html">Privacy Policy</a>
        <a href="/terms.html">Terms of Use</a>
      </nav>
    </footer>
  );
}
