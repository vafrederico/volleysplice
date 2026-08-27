import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./global.css";

if (window.location.pathname !== "/") {
  window.history.replaceState(
    null,
    "",
    `/${window.location.search}${window.location.hash}`,
  );
}

document.title = "VolleyCut";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
