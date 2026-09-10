import { useEffect, useMemo, useState } from "react";
import { AlertCircle, ExternalLink, Loader2, Play, Square } from "lucide-react";
import { runTraining, stopTraining, uploadDatasets } from "../../api";
import { useProject } from "../../state/ProjectContext";
import type { DatasetUploadResponse } from "../../types";
import { buildTrainingPayload } from "../../utils/projectPayload";
import { buildPromptTemplate, unlearningMethodLabels } from "../../defaults";
import JobProgress from "../ui/JobProgress";

export default function RunningStage() {
  const { project, setDatasetUpload, updateRun, markStageCompleted } = useProject();
  const [busy, setBusy] = useState<"train" | null>(null);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [configUrl, setConfigUrl] = useState<string | null>(null);

  const summary = useMemo(
    () => ({
      dataset:
        project.data.uploadResponse?.forget_set_path ??
        project.data.preparedForgetFile?.name ??
        project.data.forgetFile?.name ??
        "Not selected",
      retain:
        project.data.uploadResponse?.retain_set_path ??
        project.data.preparedRetainFile?.name ??
        project.data.retainFile?.name ??
        "None",
      model: project.model.modelName,
      method: project.model.method,
      unlearningMethod: unlearningMethodLabels[project.hyperparameters.unlearningMethod],
      schedule:
        project.hyperparameters.stepMode === "max_steps"
          ? `${project.hyperparameters.maxSteps} max steps`
          : `${project.hyperparameters.epochs} epochs`
    }),
    [project]
  );

  const orchestratorConfig =
    project.run.training?.orchestrator_config ?? project.run.config?.orchestrator_config ?? null;

  useEffect(() => {
    if (!orchestratorConfig) {
      setConfigUrl(null);
      return;
    }

    const blob = new Blob([JSON.stringify(orchestratorConfig, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    setConfigUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [orchestratorConfig]);

  async function ensureDataset(): Promise<DatasetUploadResponse> {
    if (project.data.uploadResponse) return project.data.uploadResponse;

    const forgetFile = project.data.preparedForgetFile ?? project.data.forgetFile;
    const retainFile = project.data.preparedRetainFile ?? project.data.retainFile;
    if (!forgetFile) throw new Error("The prepared forget dataset is missing.");

    const formData = new FormData();
    formData.append("forget_set", forgetFile);
    if (retainFile) formData.append("retain_set", retainFile);
    formData.append(
      "prompt_template",
      buildPromptTemplate(project.data.promptTemplate, project.model.modelName)
    );
    const response = await uploadDatasets(formData);
    setDatasetUpload(response, null);
    return response;
  }

  async function handleRun() {
    setError(null);
    setBusy("train");
    markStageCompleted("running", false);
    markStageCompleted("evaluation", false);
    updateRun({
      message: "Unlearning is running.",
      training: null,
      config: null,
      evaluationJob: null,
      evaluation: null,
      evaluationMessage: null
    });

    try {
      const dataset = await ensureDataset();
      const payload = buildTrainingPayload(project, dataset);
      const training = await runTraining(payload);

      updateRun({
        training,
        config: {
          status: "success",
          orchestrator_config: training.orchestrator_config,
          message: "Orchestrator config generated for unlearning."
        },
        message: training.status === "success"
          ? training.message ?? "Unlearning completed."
          : training.message
      });

      if (training.status === "success") {
        markStageCompleted("running", true);
      }
    } catch (runError) {
      updateRun({ message: null });
      setError(runError instanceof Error ? runError.message : "Unlearning failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handleStop() {
    setError(null);
    setStopping(true);
    try {
      const response = await stopTraining();
      updateRun({ message: response.message });
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : "Could not stop unlearning.");
    } finally {
      setStopping(false);
    }
  }

  return (
    <section className="stage-panel">
      <div className="stage-heading">
        <span className="stage-kicker">Stage 5</span>
        <h1>Unlearning</h1>
        <p>Review the configuration and create the unlearnt model. Evaluation follows as a separate step.</p>
      </div>

      {error && (
        <div className="notice error" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <section className="summary-grid">
        <div><span>Forget set</span><strong>{summary.dataset}</strong></div>
        <div><span>Retain set</span><strong>{summary.retain}</strong></div>
        <div><span>Model</span><strong>{summary.model}</strong></div>
        <div><span>Model method</span><strong>{summary.method}</strong></div>
        <div><span>Unlearning method</span><strong>{summary.unlearningMethod}</strong></div>
        <div><span>Schedule</span><strong>{summary.schedule}</strong></div>
        <div><span>Learning rate</span><strong>{project.hyperparameters.learningRate}</strong></div>
      </section>

      <div className="run-actions">
        {busy === "train" ? (
          <button className="danger-button" type="button" onClick={handleStop} disabled={stopping}>
            {stopping ? <Loader2 className="spin" size={17} /> : <Square size={15} fill="currentColor" />}
            {stopping ? "Stopping" : "Stop unlearning"}
          </button>
        ) : (
          <button className="primary-button" type="button" onClick={handleRun} disabled={busy !== null}>
            <Play size={17} />
            Unlearn
          </button>
        )}

        {configUrl && (
          <a className="secondary-button button-link" href={configUrl} target="_blank" rel="noreferrer">
            <ExternalLink size={16} />
            Open orchestrator config
          </a>
        )}
      </div>

      {project.run.message && <div className="notice info">{project.run.message}</div>}

      {busy === "train" && (
        <JobProgress
          title="Model unlearning"
          status={stopping ? "cancelling" : "running"}
          message={stopping ? "Stopping the active unlearning process…" : "Unlearning is running on the selected GPU."}
          progress={[{
            stage: stopping ? "stopping" : "training",
            message: stopping ? "Waiting for the backend process to stop safely." : "The training process is active. Progress continues while the terminal job is running.",
            timestamp: ""
          }]}
        />
      )}

    </section>
  );
}
