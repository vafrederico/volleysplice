import type { VolleyCutProject } from "@/lib/project-store";
import { runtimeAssetUrl } from "@/lib/runtime-assets";

import styles from "./ProjectHeader.module.css";

type ProjectHeaderProps = {
  projects: VolleyCutProject[];
  selectedProjectId: string | null;
  queueLabel: string | null;
  onSelectProject: (projectId: string | null) => void;
  onDeleteProject: () => void;
};

function projectStatus(project: VolleyCutProject): string {
  switch (project.status) {
    case "ready":
      return "Ready";
    case "analyzing":
      return "Processing";
    case "queued":
      return "Queued";
    case "waiting":
      return "Needs source";
    case "error":
      return "Stopped";
  }
}

export function ProjectHeader({
  projects,
  selectedProjectId,
  queueLabel,
  onSelectProject,
  onDeleteProject,
}: ProjectHeaderProps) {
  return (
    <header className={styles.header}>
      <span className={styles.brand}>
        {/* biome-ignore lint/performance/noImgElement: This standalone Vite app ships a local pre-sized logo without an image optimizer. */}
        <img src={runtimeAssetUrl("volleycut-logo.png")} alt="VolleyCut" />
        <span>LOCAL CUT</span>
      </span>
      <div className={styles.projectControls}>
        {queueLabel && (
          <span className={styles.queueState}>
            <i />
            {queueLabel}
          </span>
        )}
        <label className={styles.projectSelect}>
          <span>Project</span>
          <select
            aria-label="Selected project"
            value={selectedProjectId ?? "__new__"}
            onChange={(event) =>
              onSelectProject(
                event.currentTarget.value === "__new__"
                  ? null
                  : event.currentTarget.value,
              )
            }
          >
            <option value="__new__">＋ Start a new project…</option>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>
                {project.source.name} · {projectStatus(project)} · {project.id}
              </option>
            ))}
          </select>
        </label>
        {selectedProjectId && (
          <button
            className={styles.deleteButton}
            type="button"
            onClick={onDeleteProject}
          >
            Delete
          </button>
        )}
      </div>
    </header>
  );
}
