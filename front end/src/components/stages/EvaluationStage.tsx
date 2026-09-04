import { useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2, Play, Square } from "lucide-react";
import { cancelEvaluation, getEvaluationStatus, startEvaluation } from "../../api";
import type { EvaluationJobStatus, EvaluationRequest } from "../../types";
import { useProject } from "../../state/ProjectContext";
import EvaluationDashboard from "../evaluation/EvaluationDashboard";

export default function EvaluationStage() {
  const { project, updateRun, markStageCompleted } = useProject();
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const training = project.run.training;
  const job = project.run.evaluationJob;
  const isRunning = job?.status === "queued" || job?.status === "running" || job?.status === "cancelling";
  const activeJobId = isRunning && job ? job.job_id : null;

  const hasRetainSet = useMemo(() => {
    if (!training || training.status !== "success") return false;
    const dataset = training.orchestrator_config.dataset as Record<string, unknown> | undefined;
    return Boolean(dataset?.retain_set_path);
  }, [training]);

  useEffect(() => {
    if (!activeJobId) return;
    const jobId = activeJobId;
    let cancelled = false;
    let timer: number | undefined;

    async function poll() {
      try {
        const status = await getEvaluationStatus(jobId);
        if (cancelled) return;
        updateRun({
          evaluationJob: status,
          evaluationMessage: status.message,
          evaluation: status.result ?? project.run.evaluation
        });
        if (status.status === "completed" && status.result) {
          markStageCompleted("evaluation", true);
          return;
        }
        if (status.status === "failed") {
          setError(status.error ?? status.message);
          return;
        }
        if (status.status === "cancelled") return;
        timer = window.setTimeout(poll, 900);
      } catch (pollError) {
        if (cancelled) return;
        setError(pollError instanceof Error ? pollError.message : "Could not read evaluation status.");
      }
    }

    poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [activeJobId]);

  async function handleStart() {
    if (!training || training.status !== "success") {
      setError("Complete unlearning before starting evaluation.");
      return;
    }
    const embeddingModelName = project.run.embeddingModelName.trim();
    if (!embeddingModelName) {
      setError("Enter a sentence-transformers model name or local path.");
      return;
    }
    if (!hasRetainSet) {
      setError("A retain set is required to compare model utility.");
      return;
    }

    setError(null);
    markStageCompleted("evaluation", false);
    const request: EvaluationRequest = {
      orchestrator_config: training.orchestrator_config,
      training_result: training.result,
      embedding_model_name: embeddingModelName,
      max_new_tokens: project.run.evaluationMaxNewTokens
    };

    try {
      const started = await startEvaluation(request);
      const initialJob: EvaluationJobStatus = {
        job_id: started.job_id,
        status: started.status,
        current_stage: "starting",
        message: started.message,
        progress: [],
        result: null,
        error: null
      };
      updateRun({
        embeddingModelName,
        evaluationJob: initialJob,
        evaluation: null,
        evaluationMessage: started.message
      });
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Evaluation could not start.");
    }
  }

  async function handleCancel() {
    if (!activeJobId) return;
    setCancelling(true);
    setError(null);
    try {
      const response = await cancelEvaluation(activeJobId);
      if (job && response.status === "cancelling") {
        updateRun({
          evaluationJob: {
            ...job,
            status: "cancelling",
            current_stage: "cancelling",
            message: response.message
          },
          evaluationMessage: response.message
        });
      }
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : "Could not cancel evaluation.");
    } finally {
      setCancelling(false);
    }
  }

  const result = project.run.evaluation;

  return (
    <section className="stage-panel evaluation-stage">
      <div className="stage-heading">
        <span className="stage-kicker">Stage 5</span>
        <h1>Evaluate unlearning</h1>
        <p>Compare the original and unlearnt models while loading only one large model at a time.</p>
      </div>

      {!hasRetainSet && (
        <div className="notice warning">
          <AlertCircle size={18} />
          <span>This combined evaluation requires both a forget set and a retain set.</span>
        </div>
      )}
      {error && <div className="notice error" role="alert"><AlertCircle size={18} /><span>{error}</span></div>}

      <section className="evaluation-config-card">
        <div className="field-grid two">
          <label className="field">
            Sentence-transformers model
            <input
              value={project.run.embeddingModelName}
              disabled={isRunning}
              placeholder="Repository name or local model path"
              onChange={(event) => updateRun({ embeddingModelName: event.target.value })}
            />
            <small>Loaded only after both language models have been removed from memory.</small>
          </label>
          <label className="field">
            Maximum generated tokens
            <input
              type="number"
              min={1}
              disabled={isRunning}
              value={project.run.evaluationMaxNewTokens}
              onChange={(event) => updateRun({ evaluationMaxNewTokens: Math.max(1, Number(event.target.value) || 1) })}
            />
            <small>Caps generation when evaluating each answer.</small>
          </label>
        </div>
        <div className="inline-actions">
          {isRunning ? (
            <button
              className="danger-button"
              type="button"
              onClick={handleCancel}
              disabled={cancelling || job?.status === "cancelling"}
            >
              {cancelling || job?.status === "cancelling"
                ? <Loader2 className="spin" size={17} />
                : <Square size={15} fill="currentColor" />}
              {job?.status === "cancelling" ? "Cancelling" : "Cancel evaluation"}
            </button>
          ) : (
            <button className="primary-button" type="button" onClick={handleStart} disabled={!hasRetainSet}>
              <Play size={17} />
              {result ? "Run evaluation again" : "Start evaluation"}
            </button>
          )}
        </div>
      </section>

      {job && (
        <section className="evaluation-progress-card" aria-live="polite">
          <div className="progress-heading">
            <div>
              <span className={`status-dot ${job.status}`} />
              <strong>{job.message}</strong>
            </div>
            <span>{job.status}</span>
          </div>
          <ol className="progress-timeline">
            {job.progress.map((event, index) => (
              <li key={`${event.timestamp}-${index}`} className={index === job.progress.length - 1 ? "current" : "done"}>
                {index === job.progress.length - 1 && isRunning
                  ? <Loader2 className="spin" size={15} />
                  : <CheckCircle2 size={15} />}
                <div><strong>{event.message}</strong><small>{event.stage.split("_").join(" ")}</small></div>
              </li>
            ))}
          </ol>
        </section>
      )}

      {result && <EvaluationDashboard result={result} />}
    </section>
  );
}
