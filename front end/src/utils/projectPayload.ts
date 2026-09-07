import type { DatasetUploadResponse, FinalTrainingConfigRequest, Project } from "../types";

function optionalText(value: string) {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function buildTrainingPayload(
  project: Project,
  dataset: DatasetUploadResponse
): FinalTrainingConfigRequest {
  const h = project.hyperparameters;

  return {
    model_name: project.model.modelName.trim(),
    adaptor_path:
      project.model.method === "adaptor" ? optionalText(project.model.adaptorPath) : null,
    hf_key: optionalText(project.model.hfKey),
    method: project.model.method,
    unlearning_method: h.unlearningMethod,
    gpu_ids: project.model.gpuIds,
    forget_set_path: dataset.forget_set_path,
    retain_set_path: dataset.retain_set_path,
    hyperparams: {
      general: {
        max_steps: h.stepMode === "max_steps" ? h.maxSteps : null,
        epochs: h.stepMode === "epochs" ? h.epochs : null,
        learning_rate: h.learningRate,
        context_length: h.contextLength
      },
      optimization: {
        batch_size: h.batchSize,
        grad_accum: h.gradAccum,
        weight_decay: h.weightDecay
      },
      schedule: {
        save_steps: h.saveSteps
      },
      memory: {
        assistant_completions_only: true
      },
      ...(project.model.method === "lora"
        ? {
            lora_settings: {
              target_modules: project.model.selectedTargets
            }
          }
        : {})
    }
  };
}
