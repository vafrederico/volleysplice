import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./global.css";

const root = createRoot(document.getElementById("root")!);

function deploymentBasePath(): string {
  const modulePath = new URL(import.meta.url).pathname;
  const assetsMarker = "/assets/";
  const assetsIndex = modulePath.lastIndexOf(assetsMarker);

  // Production entry chunks live in <base>/assets/. Vite serves this source
  // directly during development, where the app is mounted at the domain root.
  return assetsIndex >= 0 ? modulePath.slice(0, assetsIndex + 1) : "/";
}

function currentRouteSlug(): string {
  const pathname = window.location.pathname;
  const basePath = deploymentBasePath();
  const routePath = pathname.startsWith(basePath)
    ? pathname.slice(basePath.length)
    : pathname.replace(/^\/+/, "");
  const normalized = routePath.replace(/^\/+|\/+$/g, "");

  return normalized === "index.html" ? "" : normalized;
}

const slug = currentRouteSlug();

async function renderRoute() {
  if (slug) {
    window.history.replaceState(window.history.state, "", deploymentBasePath());
  }
  const { App } = await import("./App");
  document.title = "VolleyCut";
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

void renderRoute().catch((cause: unknown) => {
  console.error(cause);
  document.title = "VolleyCut route unavailable";
  root.render(
    <main style={{ padding: "3rem", fontFamily: "system-ui, sans-serif" }}>
      <h1>This VolleyCut route could not open.</h1>
      <p>Return to the main app and try again.</p>
      <a href="./">Open VolleyCut</a>
    </main>,
  );
});
