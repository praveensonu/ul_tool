import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Database,
  FileJson,
  Loader2,
  Play,
  RefreshCw,
  Settings2,
  Upload
} from "lucide-react";
import { buildConfig, checkBackend, runTraining, uploadDatasets } from "./api";
import type {
  ConfigBuildResponse,
  DatasetUploadResponse,
  FinalTrainingConfigRequest,
  LoraTarget,
  Method,
  PreviewRow,
  StepMode,
  TrainRunResponse
} from "./types";

const defaultTemplate = `<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
`;

const loraTargets: LoraTarget[] = ["q_proj", "v_proj", "k_proj", "o_proj"];
const contextOptions = [512, 1024, 2048, 4096, 8192];

function optionalText(value: string) {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function PreviewTable({ rows }: { rows: PreviewRow[] }) {
  if (rows.length === 0) {
    return <p className="muted">No preview rows returned.</p>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Question</th>
            <th>Answer</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              <td>{String(row.question ?? "")}</td>
              <td>{String(row.answer ?? "")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [backendState, setBackendState] = useState<"checking" | "ok" | "down">(
    "checking"
  );
  const [backendMessage, setBackendMessage] = useState("Checking backend");

  const [forgetSet, setForgetSet] = useState<File | null>(null);
  const [retainSet, setRetainSet] = useState<File | null>(null);
  const [promptTemplate, setPromptTemplate] = useState(defaultTemplate);
  const [dataset, setDataset] = useState<DatasetUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);

  const [modelName, setModelName] = useState("meta-llama/Llama-2-7b-hf");
  const [method, setMethod] = useState<Method>("adaptor");
  const [adaptorPath, setAdaptorPath] = useState("");
  const [hfKey, setHfKey] = useState("");
  const [gpuId, setGpuId] = useState(0);
  const [stepMode, setStepMode] = useState<StepMode>("max_steps");
  const [maxSteps, setMaxSteps] = useState(100);
  const [epochs, setEpochs] = useState(1);
  const [learningRate, setLearningRate] = useState("1e-4");
  const [contextLength, setContextLength] = useState(2048);
  const [batchSize, setBatchSize] = useState(1);
  const [gradAccum, setGradAccum] = useState(8);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [saveSteps, setSaveSteps] = useState(10);
  const [selectedTargets, setSelectedTargets] = useState<LoraTarget[]>([
    "q_proj",
    "v_proj"
  ]);

  const [config, setConfig] = useState<ConfigBuildResponse | null>(null);
  const [training, setTraining] = useState<TrainRunResponse | null>(null);
  const [busyAction, setBusyAction] = useState<"build" | "train" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refreshBackend() {
    setBackendState("checking");
    try {
      const response = await checkBackend();
      setBackendState("ok");
      setBackendMessage(response.message);
    } catch (err) {
      setBackendState("down");
      setBackendMessage(err instanceof Error ? err.message : "Backend unavailable");
    }
  }

  useEffect(() => {
    refreshBackend();
  }, []);

  const payload = useMemo<FinalTrainingConfigRequest | null>(() => {
    if (!dataset) {
      return null;
    }

    return {
      model_name: modelName.trim(),
      adaptor_path: optionalText(adaptorPath),
      hf_key: optionalText(hfKey),
      method,
      gpu_id: gpuId,
      forget_set_path: dataset.forget_set_path,
      retain_set_path: dataset.retain_set_path,
      hyperparams: {
        general: {
          max_steps: stepMode === "max_steps" ? maxSteps : null,
          epochs: stepMode === "epochs" ? epochs : null,
          learning_rate: learningRate,
          context_length: contextLength
        },
        optimization: {
          batch_size: batchSize,
          grad_accum: gradAccum,
          weight_decay: weightDecay
        },
        schedule: {
          save_steps: saveSteps
        },
        memory: {
          assistant_completions_only: true
        },
        ...(method === "lora"
          ? {
              lora_settings: {
                target_modules: selectedTargets
              }
            }
          : {})
      }
    };
  }, [
    adaptorPath,
    batchSize,
    contextLength,
    dataset,
    epochs,
    gpuId,
    gradAccum,
    hfKey,
    learningRate,
    maxSteps,
    method,
    modelName,
    saveSteps,
    selectedTargets,
    stepMode,
    weightDecay
  ]);

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setDataset(null);
    setConfig(null);
    setTraining(null);

    if (!forgetSet) {
      setError("Choose a forget dataset first.");
      return;
    }

    const formData = new FormData();
    formData.append("forget_set", forgetSet);
    if (retainSet) {
      formData.append("retain_set", retainSet);
    }
    formData.append("prompt_template", promptTemplate);

    setUploading(true);
    try {
      const response = await uploadDatasets(formData);
      setDataset(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dataset upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleBuild() {
    setError(null);
    setConfig(null);
    setTraining(null);

    if (!payload) {
      setError("Upload datasets before building the training config.");
      return;
    }

    setBusyAction("build");
    try {
      setConfig(await buildConfig(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Config build failed.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleTrain() {
    setError(null);
    setTraining(null);

    if (!payload) {
      setError("Upload datasets before running training.");
      return;
    }

    setBusyAction("train");
    try {
      setTraining(await runTraining(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed.");
    } finally {
      setBusyAction(null);
    }
  }

  const canBuild = Boolean(payload) && busyAction === null;
  const canTrain = Boolean(payload) && busyAction === null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">LLM unlearning</p>
          <h1>Ascent training console</h1>
        </div>
        <button className="status-button" type="button" onClick={refreshBackend}>
          {backendState === "checking" ? (
            <Loader2 className="spin" size={17} />
          ) : backendState === "ok" ? (
            <CheckCircle2 size={17} />
          ) : (
            <AlertCircle size={17} />
          )}
          <span>{backendMessage}</span>
          <RefreshCw size={15} />
        </button>
      </header>

      {error && (
        <div className="notice error" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <div className="workspace">
        <section className="setup-column" aria-label="Training setup">
          <form className="section-block" onSubmit={handleUpload}>
            <div className="section-heading">
              <Database size={20} />
              <div>
                <h2>Dataset</h2>
                <p>Upload question and answer files for forget-only or retain training.</p>
              </div>
            </div>

            <div className="field-grid two">
              <label className="file-field">
                <span>Forget set</span>
                <input
                  type="file"
                  accept=".csv,.json,.jsonl,.parquet"
                  onChange={(event) => setForgetSet(event.target.files?.[0] ?? null)}
                />
              </label>
              <label className="file-field">
                <span>Retain set</span>
                <input
                  type="file"
                  accept=".csv,.json,.jsonl,.parquet"
                  onChange={(event) => setRetainSet(event.target.files?.[0] ?? null)}
                />
              </label>
            </div>

            <label className="field">
              <span>Prompt template</span>
              <textarea
                value={promptTemplate}
                onChange={(event) => setPromptTemplate(event.target.value)}
                rows={9}
              />
            </label>

            <button className="primary-button" type="submit" disabled={uploading}>
              {uploading ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
              <span>{uploading ? "Uploading" : "Upload dataset"}</span>
            </button>
          </form>

          <section className="section-block" aria-label="Model setup">
            <div className="section-heading">
              <Cpu size={20} />
              <div>
                <h2>Model</h2>
                <p>Choose the load mode, GPU, and optional adapter credentials.</p>
              </div>
            </div>

            <div className="field-grid two">
              <label className="field">
                <span>Model name or path</span>
                <input
                  value={modelName}
                  onChange={(event) => setModelName(event.target.value)}
                />
              </label>
              <label className="field">
                <span>GPU id</span>
                <input
                  type="number"
                  min={0}
                  value={gpuId}
                  onChange={(event) => setGpuId(Number(event.target.value))}
                />
              </label>
            </div>

            <div className="segmented" aria-label="Training method">
              {(["full", "lora", "adaptor"] as Method[]).map((item) => (
                <button
                  key={item}
                  type="button"
                  className={method === item ? "active" : ""}
                  onClick={() => setMethod(item)}
                >
                  {item}
                </button>
              ))}
            </div>

            <div className="field-grid two">
              <label className="field">
                <span>Adapter path</span>
                <input
                  value={adaptorPath}
                  onChange={(event) => setAdaptorPath(event.target.value)}
                  placeholder="/path/to/adapter"
                />
              </label>
              <label className="field">
                <span>HF token</span>
                <input
                  value={hfKey}
                  onChange={(event) => setHfKey(event.target.value)}
                  type="password"
                  placeholder="Optional"
                />
              </label>
            </div>

            {method === "lora" && (
              <div className="check-row" aria-label="LoRA target modules">
                {loraTargets.map((target) => (
                  <label key={target}>
                    <input
                      type="checkbox"
                      checked={selectedTargets.includes(target)}
                      onChange={(event) => {
                        setSelectedTargets((current) =>
                          event.target.checked
                            ? [...current, target]
                            : current.filter((item) => item !== target)
                        );
                      }}
                    />
                    <span>{target}</span>
                  </label>
                ))}
              </div>
            )}
          </section>

          <section className="section-block" aria-label="Hyperparameters">
            <div className="section-heading">
              <Settings2 size={20} />
              <div>
                <h2>Hyperparameters</h2>
                <p>Set the training schedule and memory-sensitive batch controls.</p>
              </div>
            </div>

            <div className="segmented compact" aria-label="Step mode">
              {(["max_steps", "epochs"] as StepMode[]).map((item) => (
                <button
                  key={item}
                  type="button"
                  className={stepMode === item ? "active" : ""}
                  onClick={() => setStepMode(item)}
                >
                  {item === "max_steps" ? "Max steps" : "Epochs"}
                </button>
              ))}
            </div>

            <div className="field-grid three">
              {stepMode === "max_steps" ? (
                <label className="field">
                  <span>Max steps</span>
                  <input
                    type="number"
                    min={1}
                    value={maxSteps}
                    onChange={(event) => setMaxSteps(Number(event.target.value))}
                  />
                </label>
              ) : (
                <label className="field">
                  <span>Epochs</span>
                  <input
                    type="number"
                    min={1}
                    value={epochs}
                    onChange={(event) => setEpochs(Number(event.target.value))}
                  />
                </label>
              )}
              <label className="field">
                <span>Learning rate</span>
                <input
                  value={learningRate}
                  onChange={(event) => setLearningRate(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Context length</span>
                <select
                  value={contextLength}
                  onChange={(event) => setContextLength(Number(event.target.value))}
                >
                  {contextOptions.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Batch size</span>
                <input
                  type="number"
                  min={1}
                  value={batchSize}
                  onChange={(event) => setBatchSize(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Grad accum</span>
                <input
                  type="number"
                  min={1}
                  value={gradAccum}
                  onChange={(event) => setGradAccum(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Save steps</span>
                <input
                  type="number"
                  min={1}
                  value={saveSteps}
                  onChange={(event) => setSaveSteps(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Weight decay</span>
                <input
                  type="number"
                  min={0}
                  step="0.001"
                  value={weightDecay}
                  onChange={(event) => setWeightDecay(Number(event.target.value))}
                />
              </label>
            </div>
          </section>
        </section>

        <aside className="result-column" aria-label="Run controls and output">
          <section className="action-strip">
            <button
              className="secondary-button"
              type="button"
              disabled={!canBuild}
              onClick={handleBuild}
            >
              {busyAction === "build" ? (
                <Loader2 className="spin" size={17} />
              ) : (
                <FileJson size={17} />
              )}
              <span>Build config</span>
            </button>
            <button
              className="primary-button"
              type="button"
              disabled={!canTrain}
              onClick={handleTrain}
            >
              {busyAction === "train" ? (
                <Loader2 className="spin" size={17} />
              ) : (
                <Play size={17} />
              )}
              <span>{busyAction === "train" ? "Running" : "Run training"}</span>
            </button>
          </section>

          <section className="result-panel">
            <h2>Dataset preview</h2>
            {dataset ? (
              <>
                <div className="metrics">
                  <span>{dataset.forget_rows} forget rows</span>
                  <span>
                    {dataset.has_retain_set
                      ? `${dataset.retain_rows} retain rows`
                      : "forget-only"}
                  </span>
                </div>
                <p className="path-text">{dataset.forget_set_path}</p>
                <PreviewTable rows={dataset.forget_preview} />
              </>
            ) : (
              <p className="muted">Upload a dataset to inspect processed rows.</p>
            )}
          </section>

          <section className="result-panel">
            <h2>Orchestrator config</h2>
            <pre>
              {config
                ? JSON.stringify(config.orchestrator_config, null, 2)
                : payload
                  ? JSON.stringify(payload, null, 2)
                  : "Upload datasets to assemble a request."}
            </pre>
          </section>

          <section className="result-panel">
            <h2>Training result</h2>
            {training ? (
              <div className="result-summary">
                <div>
                  <span>Status</span>
                  <strong>{training.result.status}</strong>
                </div>
                <div>
                  <span>Run type</span>
                  <strong>{training.result.run_type}</strong>
                </div>
                <div>
                  <span>Output</span>
                  <strong>{training.result.output_dir}</strong>
                </div>
                <pre>{JSON.stringify(training.result.metrics, null, 2)}</pre>
              </div>
            ) : (
              <p className="muted">Training output appears here after `/train/run` returns.</p>
            )}
          </section>
        </aside>
      </div>
    </main>
  );
}
