import type { Metadata } from "next";
import { ProductionEditorLab } from "@/components/production-lab/production-editor-lab";

export const metadata: Metadata = {
  title: "Editor lab · VolleySplice",
  description: "Try production and compact model review workflows on the same recording.",
};

export default function EditorLabPage() {
  return <ProductionEditorLab />;
}
