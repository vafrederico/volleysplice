"use client";

import { toggleModelSelection } from "@/lib/labeling-model-selection";
import styles from "./model-breakdown-select.module.css";

export type ModelBreakdownOption = { value: string; label: string };

export function ModelBreakdownSelect({ options, selected, onChange }: {
  options: ModelBreakdownOption[];
  selected: string[];
  onChange: (values: string[]) => void;
}) {
  const selectedCount = options.filter(option => selected.includes(option.value)).length;
  return <fieldset className={styles.field}>
    <legend>Model breakdown</legend>
    <details className={styles.dropdown} onKeyDown={event => {
      if (event.key === "Escape") {
        event.stopPropagation();
        event.currentTarget.removeAttribute("open");
        event.currentTarget.querySelector("summary")?.focus();
      }
    }}>
      <summary>{selectedCount ? `${selectedCount} model ${selectedCount === 1 ? "rail" : "rails"} selected` : "Human only"}</summary>
      <div className={styles.menu}>
        <div className={styles.actions}>
          <span>Show multiple models</span>
          <button type="button" onClick={() => onChange(options.map(option => option.value))} disabled={!options.length} aria-label="Select all model rails">All</button>
          <button type="button" onClick={() => onChange([])} disabled={!selectedCount} aria-label="Clear selected model rails">Clear</button>
        </div>
        <div className={styles.options}>
          {options.map(option => <label key={option.value} className={styles.option}>
            <input type="checkbox" checked={selected.includes(option.value)} onChange={event => onChange(toggleModelSelection(selected, option.value, event.target.checked))} />
            <span>{option.label}</span>
          </label>)}
          {!options.length && <p>Load a video with model references.</p>}
        </div>
      </div>
    </details>
  </fieldset>;
}
