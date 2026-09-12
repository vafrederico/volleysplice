import { readFileSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const httpsKey = process.env.VOLLEYCUT_DEV_HTTPS_KEY;
const httpsCertificate = process.env.VOLLEYCUT_DEV_HTTPS_CERT;
const allowedHosts = (process.env.VOLLEYCUT_DEV_ORIGINS ?? "")
  .split(",")
  .map((host) => host.trim())
  .filter(Boolean);

export default defineConfig({
  // Relative output works at a domain root, a GitHub Pages subpath, or from
  // any ordinary static host without a deployment-specific rebuild.
  base: "./",
  plugins: [react()],
  server: {
    allowedHosts,
    ...(httpsKey && httpsCertificate
      ? {
          https: {
            key: readFileSync(httpsKey),
            cert: readFileSync(httpsCertificate),
          },
        }
      : {}),
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    target: "es2022",
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 700,
  },
});
