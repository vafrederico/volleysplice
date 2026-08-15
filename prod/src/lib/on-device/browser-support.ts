export function isMacSafariBrowser(
  userAgent = typeof navigator === "undefined" ? "" : navigator.userAgent,
  platform = typeof navigator === "undefined" ? "" : navigator.platform,
  maxTouchPoints = typeof navigator === "undefined"
    ? 0
    : navigator.maxTouchPoints,
): boolean {
  const ipadInDesktopMode = platform === "MacIntel" && maxTouchPoints > 1;

  return (
    !ipadInDesktopMode &&
    /Macintosh/.test(userAgent) &&
    !/Mobile\//.test(userAgent) &&
    /Version\/\d/.test(userAgent) &&
    /Safari\//.test(userAgent) &&
    !/(?:Chrome|Chromium|CriOS|Edg|OPR|Firefox|FxiOS)\//.test(userAgent)
  );
}
