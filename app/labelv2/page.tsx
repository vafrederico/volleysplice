import type { Metadata } from "next";

import { LabelingEditor } from "@/components/labeling-editor";

export const metadata: Metadata = {
  title: "R&D Labels v2 · VolleySplice",
  description:
    "Review human and model volleyball labels against the prepared NAS dataset.",
};

export default function LabelV2Page() {
  return <LabelingEditor variant="v2" />;
}
