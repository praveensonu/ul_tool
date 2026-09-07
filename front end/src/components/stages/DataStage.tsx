import { FormEvent, useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2, Square, Upload } from "lucide-react";
import {
  cancelDatasetExtraction,
  getDatasetExtractionStatus,
  startDatasetExtraction,
  uploadDatasets
} from "../../api";
import { useProject } from "../../state/ProjectContext";
import { buildPromptTemplate } from "../../defaults";
import type {
  DataSelectionMethod,
  DataSourceMode,
  ProjectPreviewRow
} from "../../types";
import {
  backendPreviewToRows,
  parseDatasetFile,
  rowsToJsonlFile,
  selectedRows
} from "../../utils/datasetFiles";

function PreviewTable({
  rows,
  selectedKeys,
  filterable,
  onSelectionChange
}: {
  rows: ProjectPreviewRow[];
  selectedKeys: string[];
  filterable: boolean;
  onSelectionChange: (keys: string[]) => void;
}) {
  const selected = useMemo(() => new Set(selectedKeys), [selectedKeys]);
  const allSelected = rows.length > 0 && selectedKeys.length === rows.length;

  function setAll(checked: boolean) {
    if (!filterable) return;
    onSelectionChange(checked ? rows.map((row) => row.key) : []);
  }

  function toggle(key: string, checked: boolean) {
    if (!filterable) return;
    const next = new Set(selected);
    if (checked) next.add(key);
    else next.delete(key);
    onSelectionChange(rows.filter((row) => next.has(row.key)).map((row) => row.key));
  }

  if (rows.length === 0) return <p className="muted">No preview rows available.</p>;

  return (
    <>
      <div className="preview-toolbar">
        <span>{selectedKeys.length} of {rows.length} rows selected</span>
        {filterable && (
          <div className="selection-actions">
            <button className="text-button" type="button" onClick={() => setAll(true)}>Select all</button>
            <button className="text-button" type="button" onClick={() => setAll(false)}>Clear</button>
          </div>
        )}
      </div>

      <div className="table-wrap preview-table-wrap">
        <table>
          <thead>
            <tr>
              <th>
                <label className="table-check-header">
                  <input
                    type="checkbox"
                    checked={allSelected}
                    disabled={!filterable}
                    onChange={(event) => setAll(event.target.checked)}
                  />
                  ID
                </label>
              </th>
              <th>Question</th>
              <th>Answer</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} className={selected.has(row.key) ? "" : "row-ignored"}>
                <td className="table-id-cell">
                  <label>
                    <input
                      type="checkbox"
                      checked={selected.has(row.key)}
                      disabled={!filterable}
                      onChange={(event) => toggle(row.key, event.target.checked)}
                    />
                    <span>{row.id}</span>
                  </label>
                </td>
                <td>{row.question}</td>
                <td>{row.answer}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default function DataStage() {
  const {
    project,
    updateDataInput,
    setDataPrepared,
    setPreviewSelection,
    setDatasetUpload,
    setExtractionResult,
    setExtractionJob,
    updateModel
  } = useProject();
  const [uploading, setUploading] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const data = project.data;
  const extractionJob = data.extractionJob;
  const isExtracting = extractionJob?.status === "queued" ||
    extractionJob?.status === "running" || extractionJob?.status === "cancelling";
  const activeExtractionJobId = isExtracting && extractionJob ? extractionJob.job_id : null;

  useEffect(() => {
    if (!activeExtractionJobId) return;
    const jobId = activeExtractionJobId;
    let stopped = false;
    let timer: number | undefined;

    async function poll() {
      try {
        const status = await getDatasetExtractionStatus(jobId);
        if (stopped) return;
        setExtractionJob(status);
        if (status.status === "completed" && status.result) {
          const response = status.result;
          const previewRows = backendPreviewToRows(response.forget_preview);
          setExtractionResult(response);
          updateModel({
            modelName: data.extractionModelName.trim(),
            adaptorPath: data.extractionAdaptorPath.trim(),
            method: data.extractionAdaptorPath.trim() ? "adaptor" : "full"
          });
          setDataPrepared({
            previewRows,
            selectedPreviewKeys: previewRows.map((row) => row.key),
            previewReady: previewRows.length > 0,
            previewFilterable: false
          });
          return;
        }
        if (status.status === "failed") {
          setError(status.error ?? status.message);
          return;
        }
        if (status.status === "cancelled") return;
        timer = window.setTimeout(poll, 900);
      } catch (pollError) {
        if (stopped) return;
        setError(pollError instanceof Error ? pollError.message : "Could not read extraction status.");
      }
    }

    poll();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [activeExtractionJobId]);

  function changeMode(mode: DataSourceMode) {
    updateDataInput({
      sourceMode: mode,
      forgetFile: null,
      retainFile: null,
      fullFile: null,
      poisonFile: null
    });
  }

  async function uploadToBackend(forgetFile: File, retainFile: File | null) {
    const formData = new FormData();
    formData.append("forget_set", forgetFile);
    if (retainFile) formData.append("retain_set", retainFile);
    formData.append("prompt_template", buildPromptTemplate(data.promptTemplate));

    try {
      const response = await uploadDatasets(formData);
      setDatasetUpload(response, null);
      return response;
    } catch (uploadError) {
      const message = uploadError instanceof Error ? uploadError.message : "Backend upload failed.";
      setDatasetUpload(null, message);
      return null;
    }
  }

  async function prepareInitialData() {
    if (!data.forgetFile) throw new Error("Choose a forget dataset first.");

    const forgetFile = data.forgetFile;
    const retainFile = data.retainFile;
    let previewRows: ProjectPreviewRow[] = [];
    let previewFilterable = true;

    try {
      previewRows = await parseDatasetFile(forgetFile);
    } catch {
      previewFilterable = false;
    }

    let preparedForgetFile = forgetFile;
    if (previewRows.length > 0 && previewFilterable) {
      preparedForgetFile = rowsToJsonlFile(previewRows, "forget_selected.jsonl");
    }

    setDataPrepared({
      preparedForgetFile,
      preparedRetainFile: retainFile,
      previewRows,
      selectedPreviewKeys: previewRows.map((row) => row.key),
      previewReady: previewRows.length > 0,
      previewFilterable,
      backendUploadError: null
    });

    const response = await uploadToBackend(preparedForgetFile, retainFile);

    if (previewRows.length === 0 && response) {
      const backendRows = backendPreviewToRows(response.forget_preview);
      setDataPrepared({
        previewRows: backendRows,
        selectedPreviewKeys: backendRows.map((row) => row.key),
        previewReady: backendRows.length > 0,
        previewFilterable: false,
        preparedForgetFile,
        preparedRetainFile: retainFile
      });
    }

    if (previewRows.length === 0 && !response) {
      throw new Error(
        "The dataset could not be previewed locally and the backend is unavailable. For frontend-only work, use CSV, JSON, or JSONL."
      );
    }
  }

  async function applyCurrentSelection() {
    if (!data.previewFilterable) return;
    const kept = selectedRows(data.previewRows, data.selectedPreviewKeys);
    if (kept.length === 0) throw new Error("Keep at least one row in the forget set.");

    const preparedForgetFile = rowsToJsonlFile(kept, "forget_selected.jsonl");
    setDataPrepared({ preparedForgetFile });
    await uploadToBackend(preparedForgetFile, data.preparedRetainFile ?? data.retainFile);
  }

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setUploading(true);

    try {
      if (data.sourceMode === "extract") {
        if (!data.fullFile || !data.poisonFile) {
          throw new Error("Choose both the full dataset and poison set first.");
        }
        if (!data.promptTemplate.trim()) {
          throw new Error("Enter the prompt that should precede each dataset question.");
        }
        if (!data.extractionModelName.trim()) {
          throw new Error("Enter a model name or local path.");
        }
        if (data.extractionMaxLength < 1) {
          throw new Error("Max length must be at least 1.");
        }
        if (data.forgetSize < 1 || data.retainSize < 1) {
          throw new Error("Forget and retain sample counts must both be at least 1.");
        }
        if (
          data.selectionMethod === "grace" &&
          data.graceTopN < data.forgetSize
        ) {
          throw new Error("GRACE top-n must be at least the forget sample count.");
        }
        if (
          data.selectionMethod === "grace" &&
          (data.graceNumClusters < 1 || data.graceNumClusters > data.retainSize)
        ) {
          throw new Error(
            "GRACE retain clusters must be between 1 and the retain sample count."
          );
        }

        setExtractionResult(null);
        const formData = new FormData();
        formData.append("full_dataset", data.fullFile);
        formData.append("poison_set", data.poisonFile);
        formData.append("prompt_template", buildPromptTemplate(data.promptTemplate));
        formData.append("experiment_name", project.name);
        project.model.gpuIds.forEach((id) => formData.append("gpu_ids", String(id)));
        formData.append("model_name", data.extractionModelName.trim());
        formData.append("gradient_batch_size", String(data.gradientBatchSize));
        formData.append("max_length", String(data.extractionMaxLength));
        formData.append("selection_method", data.selectionMethod);
        formData.append("forget_size", String(data.forgetSize));
        formData.append("retain_size", String(data.retainSize));
        if (data.selectionMethod === "grace") {
          formData.append("grace_top_n", String(data.graceTopN));
          formData.append("grace_num_clusters", String(data.graceNumClusters));
        }
        if (data.extractionAdaptorPath.trim()) {
          formData.append("adaptor_path", data.extractionAdaptorPath.trim());
        }
        formData.append("keep_gradients", String(data.keepGradients));

        const started = await startDatasetExtraction(formData);
        setExtractionJob({
          job_id: started.job_id,
          status: started.status,
          current_stage: "starting",
          message: started.message,
          progress: [],
          result: null,
          error: null
        });
      } else if (data.previewReady && data.previewFilterable && !data.preparedForgetFile) {
        await applyCurrentSelection();
      } else {
        await prepareInitialData();
      }
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Dataset preparation failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleCancelExtraction() {
    if (!activeExtractionJobId) return;
    setCancelling(true);
    setError(null);
    try {
      const response = await cancelDatasetExtraction(activeExtractionJobId);
      if (extractionJob && response.status === "cancelling") {
        setExtractionJob({
          ...extractionJob,
          status: "cancelling",
          current_stage: "cancelling",
          message: response.message
        });
      }
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : "Could not cancel extraction.");
    } finally {
      setCancelling(false);
    }
  }

  const uploadButtonLabel = data.sourceMode === "extract"
    ? data.extractionResponse
      ? "Extract again"
      : "Extract datasets"
    : data.previewReady && !data.preparedForgetFile
      ? "Apply selection"
      : data.previewReady
        ? "Upload again"
        : "Upload dataset";

  return (
    <section className="stage-panel">
      <div className="stage-heading">
        <span className="stage-kicker">Stage 2</span>
        <h1>Data</h1>
        <p>Upload forget/retain data directly, or extract them with RASLIK or GRACE.</p>
      </div>

      {error && (
        <div className="notice error" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <form className="stage-form" onSubmit={handleUpload}>
        <fieldset className="form-fieldset" disabled={Boolean(isExtracting) || uploading}>
        <div className="field">
          <span>Data source</span>
          <div className="segmented" role="group" aria-label="Data source mode">
            {(["upload", "extract"] as DataSourceMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className={data.sourceMode === mode ? "active" : ""}
                onClick={() => changeMode(mode)}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>

        {data.sourceMode === "upload" ? (
          <div className="field-grid two">
            <label className="field">
              <span>Forget set</span>
              <input
                type="file"
                accept=".csv,.json,.jsonl,.parquet"
                onChange={(event) => updateDataInput({ forgetFile: event.target.files?.[0] ?? null })}
              />
              {data.forgetFile && <small>Saved: {data.forgetFile.name}</small>}
            </label>

            <label className="field">
              <span>Retain set <em>optional</em></span>
              <input
                type="file"
                accept=".csv,.json,.jsonl,.parquet"
                onChange={(event) => updateDataInput({ retainFile: event.target.files?.[0] ?? null })}
              />
              {data.retainFile && <small>Saved: {data.retainFile.name}</small>}
            </label>
          </div>
        ) : (
          <div className="field-grid two">
            <label className="field">
              <span>Full dataset</span>
              <input
                type="file"
                accept=".csv,.json,.jsonl,.parquet"
                onChange={(event) => updateDataInput({ fullFile: event.target.files?.[0] ?? null })}
              />
              {data.fullFile && <small>Saved: {data.fullFile.name}</small>}
            </label>

            <label className="field">
              <span>Poison set</span>
              <input
                type="file"
                accept=".csv,.json,.jsonl,.parquet"
                onChange={(event) => updateDataInput({ poisonFile: event.target.files?.[0] ?? null })}
              />
              {data.poisonFile && <small>Saved: {data.poisonFile.name}</small>}
            </label>
          </div>
        )}

        {data.sourceMode === "extract" && (
          <>
            <div className="field">
              <span>Extraction method</span>
              <div className="segmented" role="group" aria-label="Extraction method">
                {(["raslik", "grace"] as DataSelectionMethod[]).map((method) => (
                  <button
                    key={method}
                    type="button"
                    className={data.selectionMethod === method ? "active" : ""}
                    onClick={() => updateDataInput({ selectionMethod: method })}
                  >
                    {method.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <div className="field-grid two">
              <label className="field">
                <span>Forget samples</span>
                <input
                  type="number"
                  min={1}
                  value={data.forgetSize}
                  onChange={(event) => updateDataInput({ forgetSize: Number(event.target.value) })}
                />
              </label>

              <label className="field">
                <span>Retain samples</span>
                <input
                  type="number"
                  min={1}
                  value={data.retainSize}
                  onChange={(event) => updateDataInput({ retainSize: Number(event.target.value) })}
                />
              </label>
            </div>

            {data.selectionMethod === "grace" && (
              <div className="field-grid two">
                <label className="field">
                  <span>Top-n candidate pool</span>
                  <input
                    type="number"
                    min={data.forgetSize}
                    value={data.graceTopN}
                    onChange={(event) => updateDataInput({ graceTopN: Number(event.target.value) })}
                  />
                  <small>Usually around four times the forget sample count.</small>
                </label>

                <label className="field">
                  <span>Retain clusters</span>
                  <input
                    type="number"
                    min={1}
                    max={data.retainSize}
                    value={data.graceNumClusters}
                    onChange={(event) =>
                      updateDataInput({ graceNumClusters: Number(event.target.value) })
                    }
                  />
                  <small>Samples are distributed as evenly as cluster sizes allow.</small>
                </label>
              </div>
            )}

            <label className="field">
              Gradient batch size per GPU
              <input type="number" min={1} value={data.gradientBatchSize} disabled={isExtracting}
                onChange={(event) => updateDataInput({ gradientBatchSize: Math.max(1, Math.floor(Number(event.target.value) || 1)) })} />
              <small>Larger batches need more GPU memory. Set to 1 if memory is limited or the model does not support batched gradients.</small>
            </label>
            <div className="field-grid three">
              <label className="field">
                <span>Model name or local path</span>
                <input
                  value={data.extractionModelName}
                  onChange={(event) =>
                    updateDataInput({ extractionModelName: event.target.value })
                  }
                  placeholder="meta-llama/..."
                />
              </label>

              <label className="field">
                <span>Max length</span>
                <input
                  type="number"
                  min={1}
                  value={data.extractionMaxLength}
                  onChange={(event) =>
                    updateDataInput({ extractionMaxLength: Number(event.target.value) })
                  }
                />
              </label>

              <label className="field">
                <span>Adapter / LoRA path <em>optional</em></span>
                <input
                  value={data.extractionAdaptorPath}
                  onChange={(event) =>
                    updateDataInput({ extractionAdaptorPath: event.target.value })
                  }
                  placeholder="/path/to/adapter"
                />
              </label>
            </div>

            <label className="run-option">
              <input
                type="checkbox"
                checked={data.keepGradients}
                onChange={(event) => updateDataInput({ keepGradients: event.target.checked })}
              />
              <span>
                Keep cached gradients
                <small>When unchecked, training and poison gradients are removed after selection.</small>
              </span>
            </label>
          </>
        )}

        <label className="field prompt-field">
          <span>Prompt</span>

          <textarea
            rows={3}
            value={data.promptTemplate}
            placeholder="Instruction to place before each dataset question..."
            onChange={(event) =>
              updateDataInput({
                promptTemplate: event.target.value
              })
            }
          />

          <small>
            The uploaded row's question is appended and the complete model prompt is reconstructed automatically.
          </small>
        </label>
        </fieldset>

        <div className="inline-actions">
          {isExtracting ? (
            <button
              className="danger-button"
              type="button"
              onClick={handleCancelExtraction}
              disabled={cancelling || extractionJob?.status === "cancelling"}
            >
              {cancelling || extractionJob?.status === "cancelling"
                ? <Loader2 className="spin" size={17} />
                : <Square size={15} fill="currentColor" />}
              {extractionJob?.status === "cancelling" ? "Cancelling" : "Cancel extraction"}
            </button>
          ) : (
            <button className="primary-button" type="submit" disabled={uploading}>
              {uploading ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
              {uploadButtonLabel}
            </button>
          )}
        </div>
      </form>

      {extractionJob && (
        <section className="evaluation-progress-card extraction-progress-card" aria-live="polite">
          <div className="progress-heading">
            <div>
              <span className={`status-dot ${extractionJob.status}`} />
              <strong>{extractionJob.message}</strong>
            </div>
            <span>{extractionJob.status}</span>
          </div>
          <ol className="progress-timeline">
            {extractionJob.progress.map((event, index) => (
              <li
                key={`${event.timestamp}-${index}`}
                className={index === extractionJob.progress.length - 1 ? "current" : "done"}
              >
                {index === extractionJob.progress.length - 1 && isExtracting
                  ? <Loader2 className="spin" size={15} />
                  : <CheckCircle2 size={15} />}
                <div>
                  <strong>{event.message}</strong>
                  <small>{event.stage.split("_").join(" ")}</small>
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}

      {data.backendUploadError && data.previewReady && (
        <div className="notice warning">
          <AlertCircle size={18} />
          <span>
            Preview saved locally. Backend upload is pending: {data.backendUploadError}
          </span>
        </div>
      )}

      {data.uploadResponse && !data.extractionResponse && (
        <div className="notice info">
          <CheckCircle2 size={18} />
          <span>Dataset uploaded and processed by the backend.</span>
        </div>
      )}

      {data.extractionResponse && (
        <div className="notice info">
          <CheckCircle2 size={18} />
          <span>
            Extracted {data.extractionResponse.forget_rows} forget and{" "}
            {data.extractionResponse.retain_rows} retain samples with{" "}
            <strong>{data.extractionResponse.selection_method.toUpperCase()}</strong>.{" "}
            Forget set: <code>{data.extractionResponse.forget_set_path}</code>.{" "}
            Retain set: <code>{data.extractionResponse.retain_set_path}</code>.{" "}
            Cached gradients were {data.extractionResponse.gradients_retained ? "kept" : "removed"}.
          </span>
        </div>
      )}

      {data.previewReady && (
        <section className="subpanel data-preview-panel">
          <div className="subpanel-heading">
            <CheckCircle2 size={18} />
            <div>
              <h2>Dataset preview</h2>
              <p>
                {data.sourceMode === "extract"
                  ? "Preview of the extracted forget set."
                  : "Uncheck rows you want to ignore, then apply the selection before continuing."}
              </p>
            </div>
          </div>

          {!data.previewFilterable && (
            <p className="muted small-text">
              This preview came from the backend. Row filtering is disabled for this file format.
            </p>
          )}

          <PreviewTable
            rows={data.previewRows}
            selectedKeys={data.selectedPreviewKeys}
            filterable={data.previewFilterable}
            onSelectionChange={setPreviewSelection}
          />
        </section>
      )}
    </section>
  );
}
