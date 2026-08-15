import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  isChromeOnIosBrowser,
  isUnsupportedSafariBrowser,
} from "../../lib/on-device/browser-support.ts";

const rootSupportPath = new URL(
  "../../lib/on-device/browser-support.ts",
  import.meta.url,
);
const prodSupportPath = new URL(
  "../../prod/src/lib/on-device/browser-support.ts",
  import.meta.url,
);

test("local and production apps ship the same browser support policy", async () => {
  const [rootSource, prodSource] = await Promise.all([
    readFile(rootSupportPath, "utf8"),
    readFile(prodSupportPath, "utf8"),
  ]);
  assert.equal(prodSource, rootSource);
});

test("desktop Safari on macOS is unsupported", () => {
  assert.equal(
    isUnsupportedSafariBrowser(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " +
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15",
      "MacIntel",
      0,
    ),
    true,
  );
});

test("Chrome and Firefox on macOS remain supported", () => {
  assert.equal(
    isUnsupportedSafariBrowser(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " +
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    ),
    false,
  );
  assert.equal(
    isUnsupportedSafariBrowser(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:141.0) " +
        "Gecko/20100101 Firefox/141.0",
    ),
    false,
  );
});

test("Safari on iPhone and iPad is unsupported", () => {
  assert.equal(
    isUnsupportedSafariBrowser(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) " +
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Mobile/15E148 Safari/604.1",
    ),
    true,
  );
  assert.equal(
    isUnsupportedSafariBrowser(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) " +
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Mobile/15E148 Safari/604.1",
      "MacIntel",
      5,
    ),
    true,
  );
  assert.equal(
    isUnsupportedSafariBrowser(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " +
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15",
      "MacIntel",
      5,
    ),
    true,
  );
});

test("Chrome on iOS remains supported and is selected for streamed export", () => {
  const iphoneChrome =
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) " +
    "AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/140.0.0.0 Mobile/15E148 Safari/604.1";
  assert.equal(isUnsupportedSafariBrowser(iphoneChrome), false);
  assert.equal(isChromeOnIosBrowser(iphoneChrome), true);

  const ipadDesktopChrome =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " +
    "AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/140.0.0.0 Safari/605.1.15";
  assert.equal(
    isUnsupportedSafariBrowser(ipadDesktopChrome, "MacIntel", 5),
    false,
  );
  assert.equal(isChromeOnIosBrowser(ipadDesktopChrome, "MacIntel", 5), true);
});
