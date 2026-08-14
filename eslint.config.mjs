import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTypeScript from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTypeScript,
  globalIgnores([
    ".next/**",
    ".venv/**",
    "data/**",
    "out/**",
    "public/on-device/opencv.js",
    "public/on-device/opencv-worker.js",
    "next-env.d.ts",
  ]),
]);
