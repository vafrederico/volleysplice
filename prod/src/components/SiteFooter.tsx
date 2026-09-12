import styles from "./SiteFooter.module.css";

export function SiteFooter() {
  return (
    <footer className={styles.footer}>
      <div className={styles.copy}>
        <span>Your videos stay on this device. VolleySplice does not upload them.</span>
      </div>
      <nav className={styles.links} aria-label="Legal">
        <a href="https://www.volleysplice.com/privacy.html">Privacy Policy</a>
        <a href="https://www.volleysplice.com/terms.html">Terms of Use</a>
        <a href={`${import.meta.env.BASE_URL}licenses/THIRD_PARTY_NOTICES.md`}>
          Open-source notices
        </a>
      </nav>
    </footer>
  );
}
