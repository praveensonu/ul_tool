import type {
  ConfigBuildResponse,
  DatasetExtractionResponse,
  DatasetUploadResponse,
  ExtractionJobStatus,
  ExtractionStartResponse,
  EvaluationRequest,
  EvaluationJobStatus,
  EvaluationResponse,
  EvaluationStartResponse,
  FinalTrainingConfigRequest,
  TrainRunResponse,
  TrainStopResponse,
  JobCancelResponse,
  UnlearningMethodInfo
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

export async function extractDatasets(formData: FormData) {
  const response = await fetch(apiUrl("dataset/extract"), {
    method: "POST",
    body: formData
  });

  return readJson<DatasetExtractionResponse>(response);
}

export async function startDatasetExtraction(formData: FormData) {
  const response = await fetch(apiUrl("dataset/extract/start"), {
    method: "POST",
    body: formData
  });
  return readJson<ExtractionStartResponse>(response);
}

export async function getDatasetExtractionStatus(jobId: string) {
  const response = await fetch(apiUrl(`dataset/extract/status/${jobId}`));
  return readJson<ExtractionJobStatus>(response);
}

export async function cancelDatasetExtraction(jobId: string) {
  const response = await fetch(apiUrl(`dataset/extract/cancel/${jobId}`), {
    method: "POST"
  });
  return readJson<JobCancelResponse>(response);
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

export async function cancelEvaluation(jobId: string) {
  const response = await fetch(apiUrl(`evaluation/cancel/${jobId}`), {
    method: "POST"
  });
  return readJson<JobCancelResponse>(response);
}

export async function validateSentenceTransformerModel(modelName: string, hfToken?: string) {
  const value = modelName.trim();
  if (/^(?:\.{0,2}\/|~\/|\/)/.test(value)) {
    throw new Error("Local model paths cannot be verified safely from the browser. Enter a Hugging Face sentence-transformers repository name.");
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value)) {
    throw new Error("Use a valid Hugging Face repository name, for example sentence-transformers/all-MiniLM-L6-v2.");
  }

  const encodedId = value.split("/").map(encodeURIComponent).join("/");
  const headers: HeadersInit = {};
  if (hfToken?.trim()) headers.Authorization = `Bearer ${hfToken.trim()}`;

  let response: Response;
  try {
    response = await fetch(`https://huggingface.co/api/models/${encodedId}`, { headers });
  } catch {
    throw new Error("The sentence-transformers model could not be verified. Check the browser's internet access and try again.");
  }

  if (response.status === 401 || response.status === 403) {
    throw new Error("This sentence-transformers repository is private or gated. Use an accessible model and a valid Hugging Face token.");
  }
  if (response.status === 404) {
    throw new Error("Sentence-transformers model not found on Hugging Face. Check the repository name.");
  }
  if (!response.ok) {
    throw new Error(`The sentence-transformers model could not be verified (HTTP ${response.status}).`);
  }

  const metadata = await response.json() as {
    library_name?: string;
    pipeline_tag?: string;
    tags?: string[];
  };
  const tags = Array.isArray(metadata.tags) ? metadata.tags : [];
  const compatible =
    metadata.library_name === "sentence-transformers" ||
    tags.includes("sentence-transformers") ||
    metadata.pipeline_tag === "feature-extraction" ||
    metadata.pipeline_tag === "sentence-similarity";

  if (!compatible) {
    throw new Error("The repository exists, but it is not identified as a sentence-transformers/embedding model.");
  }

  return true;
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

export async function getUnlearningMethods(): Promise<UnlearningMethodInfo[]> {
  const response = await fetch(apiUrl("config/unlearning-methods"));
  const payload = await readJson<{ methods?: UnlearningMethodInfo[] }>(response);

  return Array.isArray(payload.methods) ? payload.methods : [];
}


export type GpuInfo = {
  id: number;
  name: string;
  memory_total_mb: number;
  memory_free_mb: number;
  utilization_percent: number;
  is_available: boolean;
};

export async function listGpus() {
  return readJson<{ gpus: GpuInfo[]; available_gpu_ids: number[] }>(
    await fetch(apiUrl("gpus"))
  );
}
