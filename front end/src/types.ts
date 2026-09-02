export type Method = "full" | "lora" | "adaptor";
export type UnlearningMethod = "grad_ascent" | "grad_diff" | "npo" | "dpo" | "simnpo";

export type UnlearningMethodInfo = {
  value: UnlearningMethod;
  label: string;
  requires_retain: boolean;
};
export type StepMode = "max_steps" | "epochs";
export type LoraTarget = string;
export type ProjectStage = "data" | "model" | "hyperparameters" | "running" | "evaluation";
export type DataSourceMode = "upload" | "extract";

export type BackendPreviewRow = {
  question?: string;
  answer?: string;
  [key: string]: unknown;
};

export type ProjectPreviewRow = {
  key: string;
  id: string;
  question: string;
  answer: string;
  raw: Record<string, unknown>;
};

export type DatasetUploadResponse = {
  status: string;
  forget_set_path: string;
  retain_set_path: string | null;
  forget_rows: number;
  retain_rows: number | null;
  has_retain_set: boolean;
  prompt_template: string;
  forget_preview: BackendPreviewRow[];
  retain_preview: BackendPreviewRow[] | null;
  message: string;
};

export type Hyperparams = {
  general: {
    max_steps: number | null;
    epochs: number | null;
    learning_rate: string;
    context_length: number;
  };
  optimization: {
    batch_size: number;
    grad_accum: number;
    weight_decay: number;
  };
  schedule: {
    save_steps: number;
  };
  memory: {
    assistant_completions_only: true;
  };
  lora_settings?: {
    target_modules: LoraTarget[];
  };
};

export type FinalTrainingConfigRequest = {
  model_name: string;
  adaptor_path: string | null;
  hf_key: string | null;
  method: Method;
  unlearning_method: UnlearningMethod;
  gpu_id: number;
  forget_set_path: string;
  retain_set_path: string | null;
  hyperparams: Hyperparams;
};

export type ConfigBuildResponse = {
  status: string;
  orchestrator_config: Record<string, unknown>;
  message: string;
};

type TrainResult = {
  status?: string;
  run_type?: string;
  merged_adapter?: boolean;
  output_dir?: string;
  trainable_parameters?: number;
  total_parameters?: number;
  metrics?: Record<string, unknown>;
  [key: string]: unknown;
};

export type TrainRunResponse =
  | {
      status: "success";
      orchestrator_config: Record<string, unknown>;
      result: TrainResult;
      message?: string | null;
    }
  | {
      status: "stopped";
      orchestrator_config: Record<string, unknown>;
      result: null;
      message: string;
    };

export type TrainStopResponse = {
  status: "stopped" | "idle";
  message: string;
};

export type EvaluationRequest = {
  orchestrator_config: Record<string, unknown>;
  training_result: Record<string, unknown>;
  embedding_model_name: string;
  embedding_batch_size?: number;
  max_new_tokens?: number;
};

export type ForgetQualityScores = {
  evaluated_rows: number;
  score: number;
  average_perplexity: number;
  mean_conditional_probability: number;
  mean_rouge_l: number;
  component_scores: number[];
};

export type ModelUtilityScores = {
  evaluated_rows: number;
  score: number;
  average_perplexity: number;
  mean_conditional_probability: number;
  mean_rouge_l: number;
  mean_cosine_similarity: number;
  component_scores: number[];
};

export type ModelEvaluationScores = {
  forget_quality: ForgetQualityScores;
  model_utility: ModelUtilityScores;
};

export type EvaluationResponse = {
  status: "success";
  model_path: string;
  embedding_model_name: string;
  forget_set_path: string;
  retain_set_path: string;
  pre_unlearning: ModelEvaluationScores;
  post_unlearning: ModelEvaluationScores;
  output_files: Record<string, string>;
  message: string;
};

export type EvaluationProgressEvent = {
  stage: string;
  message: string;
  timestamp: string;
};

export type EvaluationStartResponse = {
  job_id: string;
  status: "queued" | "running";
  message: string;
};

export type EvaluationJobStatus = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  current_stage: string;
  message: string;
  progress: EvaluationProgressEvent[];
  result: EvaluationResponse | null;
  error: string | null;
};

export type ProjectDataConfig = {
  sourceMode: DataSourceMode;

  // Direct upload mode.
  forgetFile: File | null;
  retainFile: File | null;

  // Extract mode: derive forget/retain from full data + poison set.
  fullFile: File | null;
  poisonFile: File | null;

  // Files actually sent to the backend after extraction / row filtering.
  preparedForgetFile: File | null;
  preparedRetainFile: File | null;

  // The full backend prompt stays fixed. The UI only shows {question}.
  promptTemplate: string;

  previewRows: ProjectPreviewRow[];
  selectedPreviewKeys: string[];
  previewReady: boolean;
  previewFilterable: boolean;

  uploadResponse: DatasetUploadResponse | null;
  backendUploadError: string | null;
};

export type ProjectModelConfig = {
  modelName: string;
  method: Method;
  adaptorPath: string;
  hfKey: string;
  gpuId: number;
  selectedTargets: LoraTarget[];
};

export type ProjectHyperparametersConfig = {
  unlearningMethod: UnlearningMethod;
  stepMode: StepMode;
  maxSteps: number;
  epochs: number;
  learningRate: string;
  contextLength: number;
  batchSize: number;
  gradAccum: number;
  weightDecay: number;
  saveSteps: number;
};

export type ProjectRunState = {
  config: ConfigBuildResponse | null;
  training: TrainRunResponse | null;
  message: string | null;
  embeddingModelName: string;
  evaluationMaxNewTokens: number;
  evaluationJob: EvaluationJobStatus | null;
  evaluation: EvaluationResponse | null;
  evaluationMessage: string | null;
};

export type CompletedStages = {
  data: boolean;
  model: boolean;
  hyperparameters: boolean;
  running: boolean;
  evaluation: boolean;
};

export type Project = {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  lastStage: ProjectStage;
  completedStages: CompletedStages;
  data: ProjectDataConfig;
  model: ProjectModelConfig;
  hyperparameters: ProjectHyperparametersConfig;
  run: ProjectRunState;
};
