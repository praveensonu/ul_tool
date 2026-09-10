import { ArrowRight, Database, Trash2 } from "lucide-react";
import type { Project } from "../../types";

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

function metric(value: number | undefined) { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—"; }

export default function ProjectCard({
  project,
  onOpen,
  onDelete
}: {
  project: Project;
  onOpen: () => void;
  onDelete: () => void;
}) {
  return (
    <article className="project-card">
      <div className="project-card-top">
        <div className="project-card-icon">
          <Database size={20} />
        </div>
        <button className="icon-button" type="button" onClick={onDelete} aria-label="Delete project">
          <Trash2 size={17} />
        </button>
      </div>

      <div>
        <h2>{project.name || "Unnamed project"}</h2>
        <p className="muted">Last stage: {project.lastStage}</p>
      </div>

      <dl className="project-card-meta">
        <div>
          <dt>Dataset</dt>
          <dd>{project.data.forgetFile?.name ?? "Not selected"}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{project.model.modelName || "Not selected"}</dd>
        </div>
        <div>
          <dt>Updated</dt>
          <dd>{formatDate(project.updatedAt)}</dd>
        </div>
      </dl>

      <div className="project-last-results">
        <span>Last evaluation</span>
        <div><small>Forget score</small><strong>{metric(project.run.evaluation?.post_unlearning.forget_quality.score)}</strong></div>
        <div><small>Utility score</small><strong>{metric(project.run.evaluation?.post_unlearning.model_utility.score)}</strong></div>
        <div><small>Forget PPL</small><strong>{metric(project.run.evaluation?.post_unlearning.forget_quality.average_perplexity)}</strong></div>
        <div><small>Retain PPL</small><strong>{metric(project.run.evaluation?.post_unlearning.model_utility.average_perplexity)}</strong></div>
      </div>

      <button className="primary-button full-width" type="button" onClick={onOpen}>
        Open project
        <ArrowRight size={17} />
      </button>
    </article>
  );
}
