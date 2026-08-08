import type { Rally } from "@/lib/edit-list";

export type CourtLine = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type ReviewAnalysis = {
  id: string;
  title: string;
  duration: number;
  width: number;
  height: number;
  sourceFilename: string;
  videoUrl: string | null;
  courtPreviewUrl: string | null;
  courtConfidence: number;
  courtSource: string;
  courtLines: CourtLine[];
  cameraStability: number;
  warnings: string[];
  rallies: Rally[];
};
