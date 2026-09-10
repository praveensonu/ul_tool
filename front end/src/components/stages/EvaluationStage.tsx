import { useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2, Play, Square } from "lucide-react";
import { cancelEvaluation, getEvaluationStatus, startEvaluation, validateSentenceTransformerModel } from "../../api";
import type { EvaluationJobStatus, EvaluationRequest } from "../../types";
import { useProject } from "../../state/ProjectContext";
import EvaluationDashboard from "../evaluation/EvaluationDashboard";
import JobProgress from "../ui/JobProgress";
import { FieldLabel, fieldHelp } from "../ui/HelpTip";

export default function EvaluationStage() {
  const { project, updateRun, markStageCompleted } = useProject();
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [embeddingValidation, setEmbeddingValidation] = useState<"idle" | "checking" | "valid">("idle");
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

  async function checkEmbeddingModel() {
    const embeddingModelName = project.run.embeddingModelName.trim();
    if (!embeddingModelName) {
      setEmbeddingValidation("idle");
      throw new Error("The sentence-transformers model is mandatory.");
    }

    setEmbeddingValidation("checking");
    try {
      await validateSentenceTransformerModel(embeddingModelName, project.model.hfKey);
      setEmbeddingValidation("valid");
      return embeddingModelName;
    } catch (validationError) {
      setEmbeddingValidation("idle");
      throw validationError;
    }
  }

  async function handleEmbeddingBlur() {
    if (!project.run.embeddingModelName.trim() || isRunning) return;
    try {
      await checkEmbeddingModel();
      setError(null);
    } catch (validationError) {
      setError(validationError instanceof Error ? validationError.message : "The sentence-transformers model is invalid.");
    }
  }

  async function handleStart() {
    if (!training || training.status !== "success") {
      setError("Complete unlearning before starting evaluation.");
      return;
    }
    if (!hasRetainSet) {
      setError("A retain set is required to compare model utility.");
      return;
    }

    setError(null);
    let embeddingModelName: string;
    try {
      embeddingModelName = await checkEmbeddingModel();
    } catch (validationError) {
      setError(validationError instanceof Error ? validationError.message : "The sentence-transformers model is invalid.");
      return;
    }
    markStageCompleted("evaluation", false);
    const request: EvaluationRequest = {
      orchestrator_config: training.orchestrator_config,
      training_result: training.result,
      embedding_model_name: embeddingModelName,
      experiment_name: project.name,
      batch_size: project.run.evaluationBatchSize,
      include_benchmarks: project.run.includeBenchmarks,
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
        <span className="stage-kicker">Stage 6</span>
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

      {result?.output_files.results_jsonl_path && <p>Results saved to <code>{result.output_files.results_jsonl_path}</code></p>}
      <section className="evaluation-config-card">
        <label className="checkbox-field">
          <input type="checkbox" checked={project.run.includeBenchmarks} disabled={isRunning}
            onChange={(event) => updateRun({ includeBenchmarks: event.target.checked })} />
          <FieldLabel help={fieldHelp.benchmarks}>Include benchmark evaluation (MMLU and GPQA)</FieldLabel>
        </label>
        <label className="field">
          <FieldLabel help={fieldHelp.evaluationBatch}>Evaluation batch size</FieldLabel>
          <input type="number" min={1} disabled={isRunning} value={project.run.evaluationBatchSize}
            onChange={(event) => updateRun({ evaluationBatchSize: Math.max(1, Math.floor(Number(event.target.value) || 1)) })} />
          <small>Used for generation, conditional probability, and perplexity. Reduce it if GPU memory is limited.</small>
        </label>
        <div className="field-grid two">
          <label className="field">
            <FieldLabel help={fieldHelp.embeddingModel}>Sentence-transformers model <em className="required-mark">required</em></FieldLabel>
            <input
              required
              aria-required="true"
              aria-invalid={Boolean(error) && embeddingValidation !== "valid"}
              value={project.run.embeddingModelName}
              disabled={isRunning}
              placeholder="e.g. sentence-transformers/all-MiniLM-L6-v2"
              onBlur={handleEmbeddingBlur}
              onChange={(event) => {
                setEmbeddingValidation("idle");
                updateRun({ embeddingModelName: event.target.value });
              }}
            />
            <small>Verified on Hugging Face before evaluation starts; loaded after both language models leave memory.</small>
            {embeddingValidation === "checking" && <span className="field-validation checking"><Loader2 className="spin" size={14} />Checking model…</span>}
            {embeddingValidation === "valid" && <span className="field-validation valid"><CheckCircle2 size={14} />Model verified</span>}
          </label>
          <label className="field">
            <FieldLabel help={fieldHelp.maxTokens}>Maximum generated tokens</FieldLabel>
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
            <button className="primary-button" type="button" onClick={handleStart} disabled={!hasRetainSet || !project.run.embeddingModelName.trim() || embeddingValidation === "checking"}>
              {embeddingValidation === "checking" ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
              {embeddingValidation === "checking" ? "Checking model" : result ? "Run evaluation again" : "Start evaluation"}
            </button>
          )}
        </div>
      </section>

      {job && <JobProgress title="Model evaluation" status={job.status} message={job.message} progress={job.progress} splitEvaluation />}

      {result && <EvaluationDashboard result={result} />}
    </section>
  );
}
