export const SIDE_SWITCH_PROPOSAL_MODELS = ["v5", "v5-state", "v6"] as const;

export type SideSwitchProposalModel =
  (typeof SIDE_SWITCH_PROPOSAL_MODELS)[number];

export type SideSwitchModelProposal = {
  modelId: SideSwitchProposalModel;
  evaluationEventId: string;
  gapOrder: number;
  score: number;
  selected: boolean;
};

export type SideSwitchProposalLayer = {
  modelId: SideSwitchProposalModel;
  label: string;
  detail: string;
  evaluationKind: string;
  evaluationCreatedAt: string;
  evaluationFilename: string;
  evaluationSha256: string;
  featureFilename: string;
  featureSha256: string;
  modelFilename: string | null;
  modelSha256: string | null;
  threshold: number;
  candidateMargin: number;
  evaluatedEvents: number;
  selectedEvents: number;
  attachedEvents: number;
  selectedAttachedEvents: number;
};

export type SideSwitchProposalLoadError = {
  modelId: SideSwitchProposalModel;
  message: string;
};

export type SideSwitchProposalBundle = {
  layers: SideSwitchProposalLayer[];
  byEventId: Record<
    string,
    Partial<Record<SideSwitchProposalModel, SideSwitchModelProposal>>
  >;
  errors: SideSwitchProposalLoadError[];
};

export const EMPTY_SIDE_SWITCH_PROPOSAL_BUNDLE: SideSwitchProposalBundle = {
  layers: [],
  byEventId: {},
  errors: [],
};
