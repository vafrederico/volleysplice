type ModelIdentity = { modelId: string; research?: unknown };

export function initialModelSelection(production: ModelIdentity | null, experiments: readonly ModelIdentity[]): string[] {
  const research = experiments.find(reference => reference.research);
  return [
    ...(production ? ["production"] : []),
    ...(research ? [`model:${research.modelId}`] : []),
  ];
}

export function toggleModelSelection(selected: readonly string[], value: string, checked: boolean): string[] {
  return checked ? [...new Set([...selected, value])] : selected.filter(candidate => candidate !== value);
}

export function selectedModelReferences<T extends ModelIdentity>(production: T | null, experiments: readonly T[], selected: readonly string[]): T[] {
  return [
    ...(production && selected.includes("production") ? [production] : []),
    ...experiments.filter(reference => selected.includes(`model:${reference.modelId}`)),
  ];
}
