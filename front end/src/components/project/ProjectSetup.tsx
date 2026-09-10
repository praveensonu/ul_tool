import { ArrowRight, FileInput, Sparkles } from "lucide-react";
import { useState } from "react";
import { useProject } from "../../state/ProjectContext";
import type { DataSourceMode } from "../../types";
import HelpTip, { fieldHelp } from "../ui/HelpTip";

const options: Array<{ mode: DataSourceMode; title: string; description: string; preview: string[] }> = [
  { mode: "upload", title: "Upload datasets", description: "Start with a mandatory forget dataset and an optional retain dataset that are already prepared.", preview: ["GPUs", "Data", "Model", "Hyperparameters", "Unlearning", "Evaluation"] },
  { mode: "extract", title: "Extract datasets", description: "Derive forget and retain sets from full data using RASLIK or GRACE.", preview: ["GPUs", "Data + model", "Hyperparameters", "Unlearning", "Evaluation"] }
];

export default function ProjectSetup() {
  const { project, renameProject, finishSetup } = useProject();
  const [step, setStep] = useState<"name" | "source">("name");
  const displayName = project.name === "Unnamed project" ? "" : project.name;

  return (
    <div className="setup-overlay">
      <section className="setup-card" aria-labelledby="setup-title">
        <div className="setup-progress"><span className="active">1</span><i /><span className={step === "source" ? "active" : ""}>2</span></div>
        {step === "name" ? (
          <>
            <p className="eyebrow">New unlearning project</p>
            <h1 id="setup-title">Name your project</h1>
            <p className="muted">Give this run a clear name. You can rename it later.</p>
            <label className="field setup-name-field">
              <span className="field-label">Project name <HelpTip text={fieldHelp.projectName} /></span>
              <input autoFocus value={displayName} placeholder="Unnamed project" onChange={(event) => renameProject(event.target.value || "Unnamed project")} onKeyDown={(event) => event.key === "Enter" && setStep("source")} />
            </label>
            <button className="primary-button setup-next" type="button" onClick={() => setStep("source")}>Choose data source <ArrowRight size={17} /></button>
          </>
        ) : (
          <>
            <p className="eyebrow">New unlearning project</p>
            <div className="title-with-help"><h1 id="setup-title">How will you prepare the data?</h1><HelpTip text={fieldHelp.sourceMode} /></div>
            <p className="muted">This determines the stages in your project pipeline.</p>
            <div className="source-card-grid">
              {options.map((option) => (
                <button className="source-choice-card" type="button" key={option.mode} onClick={() => finishSetup(option.mode)}>
                  <span className="source-choice-icon">{option.mode === "upload" ? <FileInput /> : <Sparkles />}</span>
                  <strong>{option.title}</strong>
                  <p>{option.description}</p>
                  <span className="pipeline-preview">{option.preview.map((item, index) => <span key={item}>{index > 0 && <i>→</i>}{item}</span>)}</span>
                </button>
              ))}
            </div>
            <button className="text-button" type="button" onClick={() => setStep("name")}>Back to project name</button>
          </>
        )}
      </section>
    </div>
  );
}
