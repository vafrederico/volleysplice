import { networkInterfaces } from "node:os";

function localInterfaceOrigins() {
  return Object.values(networkInterfaces())
    .flatMap((addresses) => addresses ?? [])
    .filter((address) => address.family === "IPv4" && !address.internal)
    .map((address) => address.address);
}

function configuredDevOrigins() {
  return (process.env.VOLLEYCUT_DEV_ORIGINS ?? "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);
}

const nextConfig = {
  // `next dev --hostname 0.0.0.0` otherwise rejects HMR and dev chunks
  // requested through the host's real LAN address. Values are hostnames only.
  allowedDevOrigins: [...new Set([...localInterfaceOrigins(), ...configuredDevOrigins()])],
};

export default nextConfig;
