"use client";

import { useEffect, useState } from "react";
import Workspace from "./workspace";

export function ProductionEditorLab() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted ? <Workspace /> : <p style={{ padding: 24 }}>Opening the editor lab…</p>;
}
