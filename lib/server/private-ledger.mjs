import { readFileSync } from "node:fs";

// Server/research tooling only. Never import this module into a client component.
export function privateValue(index) {
  const platform = process.platform === "win32" ? "WINDOWS" : "POSIX";
  const location = process.env[`VOLLEYCUT_PRIVATE_LEDGER_${platform}`] || process.env.VOLLEYCUT_PRIVATE_LEDGER;
  if (!location) throw new Error("Set VOLLEYCUT_PRIVATE_LEDGER to the private research ledger");
  const ledger = JSON.parse(readFileSync(location, "utf8"));
  const value = ledger.privateValues?.[index];
  if (typeof value !== "string" || !value) throw new Error(`Missing private ledger index: ${index}`);
  return value;
}
