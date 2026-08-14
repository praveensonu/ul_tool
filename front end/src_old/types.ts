export type Method = "full" | "lora" | "adaptor";
export type StepMode = "max_steps" | "epochs";
export type LoraTarget = "q_proj" | "v_proj" | "k_proj" | "o_proj";

export type PreviewRow = {
  question?: string;
  answer?: string;
  [key: string]: unknown;
};

export type DatasetUploadResponse = {
  status: string;
  forget_set_path: string;
  retain_set_path: string | null;
  forget_rows: number;
  retain_rows: number | null;
  has_retain_set: boolean;
  prompt_template: string;
  forget_preview: PreviewRow[];
  retain_preview: PreviewRow[] | null;
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
  status: string;
  run_type: string;
  merged_adapter: boolean;
  output_dir: string;
  trainable_parameters: number;
  total_parameters: number;
  metrics: Record<string, unknown>;
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
