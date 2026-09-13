import type { VolleySpliceProject } from "@/lib/project-store";
import { runtimeAssetUrl } from "@/lib/runtime-assets";
import type { DesignExportJob } from "@/designs/useDesignReview";

import styles from "./ProjectHeader.module.css";

type ProjectHeaderProps = {
  projects: VolleySpliceProject[];
  exportJobs: DesignExportJob[];
  selectedProjectId: string | null;
  queueLabel: string | null;
  onSelectProject: (projectId: string | null) => void;
  onDeleteProject: () => void;
};

function projectStatus(
  project: VolleySpliceProject,
  exportJob: DesignExportJob | undefined,
): string {
  if (project.status === "ready" && exportJob) {
    switch (exportJob.status) {
      case "queued":
        return "Export queued";
      case "exporting":
        return `Exporting ${exportJob.progress}%`;
      case "saved":
        return "Exported";
      case "error":
        return "Export stopped";
    }
  }
  switch (project.status) {
    case "ready":
      return project.lastExportedAt ? "Exported" : "Ready";
    case "analyzing":
      return "Analyzing";
    case "queued":
      return "Queued";
    case "waiting":
      return "Needs original video";
    case "error":
      return "Stopped";
  }
}

export function ProjectHeader({
  projects,
  exportJobs,
  selectedProjectId,
  queueLabel,
  onSelectProject,
  onDeleteProject,
}: ProjectHeaderProps) {
  const exportJobsByProject = new Map(
    exportJobs.map((job) => [job.projectId, job] as const),
  );
  return (
    <header className={styles.header} data-tour="editor-header">
      <span className={styles.brand}>
        {/* biome-ignore lint/performance/noImgElement: This standalone Vite app ships a local pre-sized logo without an image optimizer. */}
        <img
          src={runtimeAssetUrl("volleysplice-icon-transparent.png")}
          alt=""
        />
        <strong className={styles.wordmark}>
          volley<span>splice</span>
        </strong>
        <span className={styles.productLabel}>VIDEO EDITOR</span>
      </span>
      <div className={styles.projectControls}>
        {queueLabel && (
          <span className={styles.queueState}>
            <i />
            {queueLabel}
          </span>
        )}
        <label className={styles.projectSelect}>
          <span>Current project</span>
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
            <option value="__new__">＋ Start a new video…</option>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>
                {project.source.name} · {projectStatus(
                  project,
                  exportJobsByProject.get(project.id),
                )}
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
