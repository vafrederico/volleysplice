import styles from "./ScoreVisibilitySelect.module.css";

type Props = {
  fade: boolean;
  onChange: (fade: boolean) => void;
};

export function ScoreVisibilitySelect({ fade, onChange }: Props) {
  return (
    <label className={styles.setting}>
      <strong>Score visibility</strong>
      <select value={fade ? "fade" : "always"} onChange={(event) => onChange(event.currentTarget.value === "fade")}>
        <option value="always">Always visible</option>
        <option value="fade">Fade in and out</option>
      </select>
      <small>Fading follows the point timeline. The timeline always fades in and out.</small>
    </label>
  );
}
