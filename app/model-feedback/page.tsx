import type { Metadata } from "next";

import { ModelFeedbackImporter } from "./model-feedback-importer";

export const metadata: Metadata = {
  title: "Import model feedback · VolleyCut",
  description:
    "Inspect and permanently link production web and Android model-feedback bundles.",
};

export const dynamic = "force-dynamic";

export default async function ModelFeedbackPage({
  searchParams,
}: {
  searchParams: Promise<{ import?: string | string[] }>;
}) {
  const query = await searchParams;
  return (
    <ModelFeedbackImporter
      initialImportId={typeof query.import === "string" ? query.import : null}
    />
  );
}
