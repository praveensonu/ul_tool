import { FormEvent, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2, Upload } from "lucide-react";
import { uploadDatasets } from "../../api";
import { useProject } from "../../state/ProjectContext";
import { defaultTemplate } from "../../defaults";
import type { DataSourceMode, ProjectPreviewRow } from "../../types";
import {
  backendPreviewToRows,
  extractForgetAndRetain,
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
    setDatasetUpload
  } = useProject();
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const data = project.data;

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
    const fullPrompt = defaultTemplate.replace(
        "{question}",
        data.promptTemplate.trim()
      );

    formData.append("prompt_template", fullPrompt);

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
    let forgetFile: File;
    let retainFile: File | null;
    let previewRows: ProjectPreviewRow[] = [];
    let previewFilterable = true;

    if (data.sourceMode === "extract") {
      if (!data.fullFile || !data.poisonFile) {
        throw new Error("Choose both the full dataset and poison set first.");
      }
      const extracted = await extractForgetAndRetain(data.fullFile, data.poisonFile);
      forgetFile = extracted.forgetFile;
      retainFile = extracted.retainFile;
      previewRows = extracted.forgetRows;
      setDataPrepared({ forgetFile, retainFile });
    } else {
      if (!data.forgetFile) throw new Error("Choose a forget dataset first.");
      forgetFile = data.forgetFile;
      retainFile = data.retainFile;
      try {
        previewRows = await parseDatasetFile(forgetFile);
      } catch {
        previewFilterable = false;
      }
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
      if (data.previewReady && data.previewFilterable && !data.preparedForgetFile) {
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

  const uploadButtonLabel = data.previewReady && !data.preparedForgetFile
    ? "Apply selection"
    : data.previewReady
      ? "Upload again"
      : "Upload dataset";

  return (
    <section className="stage-panel">
      <div className="stage-heading">
        <span className="stage-kicker">Stage 1</span>
        <h1>Data</h1>
        <p>Upload forget/retain data directly, or extract them from a full dataset and poison set.</p>
      </div>

      {error && (
        <div className="notice error" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <form className="stage-form" onSubmit={handleUpload}>
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
                accept=".csv,.json,.jsonl"
                onChange={(event) => updateDataInput({ fullFile: event.target.files?.[0] ?? null })}
              />
              {data.fullFile && <small>Saved: {data.fullFile.name}</small>}
            </label>

            <label className="field">
              <span>Poison set</span>
              <input
                type="file"
                accept=".csv,.json,.jsonl"
                onChange={(event) => updateDataInput({ poisonFile: event.target.files?.[0] ?? null })}
              />
              {data.poisonFile && <small>Saved: {data.poisonFile.name}</small>}
            </label>
          </div>
        )}

        <label className="field prompt-field">
          <span>Question</span>

          <textarea
            rows={3}
            value={data.promptTemplate}
            placeholder="Type the question here..."
            onChange={(event) =>
              updateDataInput({
                promptTemplate: event.target.value
              })
            }
          />

          <small>
            Only enter the question. The complete prompt is reconstructed automatically.
          </small>
        </label>

        <div className="inline-actions">
          <button className="primary-button" type="submit" disabled={uploading}>
            {uploading ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
            {uploadButtonLabel}
          </button>
        </div>
      </form>

      {data.backendUploadError && data.previewReady && (
        <div className="notice warning">
          <AlertCircle size={18} />
          <span>
            Preview saved locally. Backend upload is pending: {data.backendUploadError}
          </span>
        </div>
      )}

      {data.uploadResponse && (
        <div className="notice info">
          <CheckCircle2 size={18} />
          <span>Dataset uploaded and processed by the backend.</span>
        </div>
      )}

      {data.previewReady && (
        <section className="subpanel data-preview-panel">
          <div className="subpanel-heading">
            <CheckCircle2 size={18} />
            <div>
              <h2>Dataset preview</h2>
              <p>Uncheck rows you want to ignore, then apply the selection before continuing.</p>
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
