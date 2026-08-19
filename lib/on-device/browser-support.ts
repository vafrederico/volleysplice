function currentUserAgent(): string {
  return typeof navigator === "undefined" ? "" : navigator.userAgent;
}

export function isAndroidBrowser(userAgent = currentUserAgent()): boolean {
  return /Android/i.test(userAgent);
}

function currentPlatform(): string {
  return typeof navigator === "undefined" ? "" : navigator.platform;
}

function currentMaxTouchPoints(): number {
  return typeof navigator === "undefined" ? 0 : navigator.maxTouchPoints;
}

export function isIosBrowser(
  userAgent = currentUserAgent(),
  platform = currentPlatform(),
  maxTouchPoints = currentMaxTouchPoints(),
): boolean {
  return (
    /iPad|iPhone|iPod/.test(userAgent) ||
    (platform === "MacIntel" && maxTouchPoints > 1)
  );
}

export function isChromeOnIosBrowser(
  userAgent = currentUserAgent(),
  platform = currentPlatform(),
  maxTouchPoints = currentMaxTouchPoints(),
): boolean {
  return (
    /CriOS\//.test(userAgent) &&
    isIosBrowser(userAgent, platform, maxTouchPoints)
  );
}

export function isUnsupportedSafariBrowser(
  userAgent = currentUserAgent(),
  platform = currentPlatform(),
  maxTouchPoints = currentMaxTouchPoints(),
): boolean {
  const ios = isIosBrowser(userAgent, platform, maxTouchPoints);
  return (
    (ios || /Macintosh/.test(userAgent)) &&
    /Version\/\d/.test(userAgent) &&
    /Safari\//.test(userAgent) &&
    !/(?:Chrome|Chromium|CriOS|Edg|EdgiOS|OPR|OPiOS|Firefox|FxiOS)\//.test(
      userAgent,
    )
  );
}
