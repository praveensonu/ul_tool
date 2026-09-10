import { Check } from "lucide-react";
import type { Project, ProjectStage } from "../../types";
import { canAccessStage, isStageComplete } from "../../utils/projectValidation";

const steps: Array<{ id: ProjectStage; label: string }> = [
  { id: "gpu", label: "GPUs" },
  { id: "data", label: "Data" },
  { id: "model", label: "Model" },
  { id: "hyperparameters", label: "Hyperparameters" },
  { id: "running", label: "Unlearning" },
  { id: "evaluation", label: "Evaluation" }
];

export default function StepNavigation({
  project,
  currentStage,
  onSelect
}: {
  project: Project;
  currentStage: ProjectStage;
  onSelect: (stage: ProjectStage) => void;
}) {
  const visibleSteps = project.data.sourceMode === "extract" ? steps.filter((step) => step.id !== "model") : steps;
  return (
    <nav className="step-navigation" aria-label="Project stages">
      {visibleSteps.map((step, index) => {
        const accessible = canAccessStage(project, step.id);
        const complete = isStageComplete(project, step.id);
        const active = step.id === currentStage;

        return (
          <button
            key={step.id}
            className={`step-button ${active ? "active" : ""} ${complete ? "complete" : ""}`}
            type="button"
            disabled={!accessible}
            onClick={() => onSelect(step.id)}
          >
            <span className="step-number">{complete ? <Check size={15} /> : index + 1}</span>
            <span>{step.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
