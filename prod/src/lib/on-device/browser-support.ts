export function isMacSafariBrowser(
  userAgent = typeof navigator === "undefined" ? "" : navigator.userAgent,
): boolean {
  return (
    /Macintosh/.test(userAgent) &&
    !/Mobile\//.test(userAgent) &&
    /Version\/\d/.test(userAgent) &&
    /Safari\//.test(userAgent) &&
    !/(?:Chrome|Chromium|CriOS|Edg|OPR|Firefox|FxiOS)\//.test(userAgent)
  );
}
