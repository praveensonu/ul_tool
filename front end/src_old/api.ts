import type {
  ConfigBuildResponse,
  DatasetUploadResponse,
  FinalTrainingConfigRequest,
  TrainRunResponse,
  TrainStopResponse
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

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
  const response = await fetch(`${API_BASE}/`);
  return readJson<{ message: string }>(response);
}

export async function uploadDatasets(formData: FormData) {
  const response = await fetch(`${API_BASE}/dataset/upload`, {
    method: "POST",
    body: formData
  });

  return readJson<DatasetUploadResponse>(response);
}

export async function buildConfig(payload: FinalTrainingConfigRequest) {
  const response = await fetch(`${API_BASE}/config/build`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  return readJson<ConfigBuildResponse>(response);
}

export async function runTraining(payload: FinalTrainingConfigRequest) {
  const response = await fetch(`${API_BASE}/train/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  return readJson<TrainRunResponse>(response);
}

export async function stopTraining() {
  const response = await fetch(`${API_BASE}/train/stop`, {
    method: "POST"
  });

  return readJson<TrainStopResponse>(response);
}
