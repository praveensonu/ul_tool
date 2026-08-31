import type {
  ConfigBuildResponse,
  DatasetUploadResponse,
  EvaluationRequest,
  EvaluationJobStatus,
  EvaluationResponse,
  EvaluationStartResponse,
  FinalTrainingConfigRequest,
  TrainRunResponse,
  TrainStopResponse
} from "./types";
import { API_BASE_URL, apiUrl } from "./apiConfig";

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((item) => item.msg || JSON.stringify(item)).join("; ")
          : `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return payload as T;
}

export async function checkBackend() {
  const response = await fetch(apiUrl("health"));
  return readJson<{ message: string }>(response);
}

export async function uploadDatasets(formData: FormData) {
  const response = await fetch(apiUrl("dataset/upload"), {
    method: "POST",
    body: formData
  });

  return readJson<DatasetUploadResponse>(response);
}

export async function buildConfig(payload: FinalTrainingConfigRequest) {
  const response = await fetch(apiUrl("config/build"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return readJson<ConfigBuildResponse>(response);
}

export async function runTraining(payload: FinalTrainingConfigRequest) {
  const response = await fetch(apiUrl("train/run"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  return readJson<TrainRunResponse>(response);
}

export async function stopTraining() {
  const response = await fetch(apiUrl("train/stop"), {
    method: "POST"
  });
  return readJson<TrainStopResponse>(response);
}

export async function startEvaluation(payload: EvaluationRequest) {
  const response = await fetch(apiUrl("evaluation/start"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return readJson<EvaluationStartResponse>(response);
}

export async function getEvaluationStatus(jobId: string) {
  const response = await fetch(apiUrl(`evaluation/status/${jobId}`));
  return readJson<EvaluationJobStatus>(response);
}

export async function runEvaluation(payload: EvaluationRequest) {
  const response = await fetch(apiUrl("evaluation/run"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return readJson<EvaluationResponse>(response);
}

export async function getLoraTargetModules(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/openapi.json`);
  const openApi = await readJson<Record<string, unknown>>(response);
  const components = openApi.components as Record<string, unknown> | undefined;
  const schemas = components?.schemas as Record<string, unknown> | undefined;
  const targetSchema = schemas?.LoraTargetModule as Record<string, unknown> | undefined;
  const values = targetSchema?.enum;

  return Array.isArray(values) ? values.filter((item): item is string => typeof item === "string") : [];
}
