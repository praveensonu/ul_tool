import type { CompletedStages, Project, ProjectStage } from "./types";

export const defaultTemplate = `<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
`;

export const defaultLoraTargets = ["q_proj", "v_proj", "k_proj", "o_proj"];

export function createDefaultProject(name = "Untitled project"): Project {
  const now = new Date().toISOString();

  return {
    id: crypto.randomUUID(),
    name,
    createdAt: now,
    updatedAt: now,
    lastStage: "data",
    completedStages: {
      data: false,
      model: false,
      hyperparameters: false,
      running: false,
      evaluation: false
    },
    data: {
      sourceMode: "upload",
      forgetFile: null,
      retainFile: null,
      fullFile: null,
      poisonFile: null,
      preparedForgetFile: null,
      preparedRetainFile: null,
      promptTemplate: "",
      previewRows: [],
      selectedPreviewKeys: [],
      previewReady: false,
      previewFilterable: true,
      uploadResponse: null,
      backendUploadError: null
    },
    model: {
      modelName: "meta-llama/Llama-2-7b-hf",
      method: "full",
      adaptorPath: "",
      hfKey: "",
      gpuId: 0,
      selectedTargets: [...defaultLoraTargets]
    },
    hyperparameters: {
      stepMode: "max_steps",
      maxSteps: 100,
      epochs: 1,
      learningRate: "1e-4",
      contextLength: 2048,
      batchSize: 1,
      gradAccum: 8,
      weightDecay: 0.01,
      saveSteps: 10
    },
    run: {
      config: null,
      training: null,
      message: null,
      embeddingModelName: "",
      evaluationMaxNewTokens: 256,
      evaluationJob: null,
      evaluation: null,
      evaluationMessage: null
    }
  };
}

function migratedCompletion(lastStage: ProjectStage, hasTraining: boolean, hasEvaluation: boolean): CompletedStages {
  const stageIndex = ["data", "model", "hyperparameters", "running", "evaluation"].indexOf(lastStage);
  return {
    data: stageIndex >= 1,
    model: stageIndex >= 2,
    hyperparameters: stageIndex >= 3,
    running: hasTraining,
    evaluation: stageIndex >= 4 && hasEvaluation
  };
}

export function normalizeProject(value: unknown): Project {
  const raw = (value ?? {}) as Partial<Project> & Record<string, unknown>;
  const base = createDefaultProject(typeof raw.name === "string" ? raw.name : "Untitled project");
  const lastStage: ProjectStage =
    raw.lastStage === "model" ||
    raw.lastStage === "hyperparameters" ||
    raw.lastStage === "running" ||
    raw.lastStage === "evaluation"
      ? raw.lastStage
      : "data";

  const rawRun = (raw.run ?? {}) as Partial<Project["run"]>;
  const rawData = (raw.data ?? {}) as Partial<Project["data"]>;
  const rawModel = (raw.model ?? {}) as Partial<Project["model"]>;
  const rawHyperparameters = (raw.hyperparameters ?? {}) as Partial<Project["hyperparameters"]>;
  const completion = raw.completedStages ?? migratedCompletion(
    lastStage,
    rawRun.training !== null && rawRun.training !== undefined,
    rawRun.evaluation !== null && rawRun.evaluation !== undefined
  );

  return {
    ...base,
    ...raw,
    id: typeof raw.id === "string" ? raw.id : base.id,
    name: typeof raw.name === "string" ? raw.name : base.name,
    createdAt: typeof raw.createdAt === "string" ? raw.createdAt : base.createdAt,
    updatedAt: typeof raw.updatedAt === "string" ? raw.updatedAt : base.updatedAt,
    lastStage,
    completedStages: {
      ...base.completedStages,
      ...completion
    },
    data: {
      ...base.data,
      ...rawData,
      sourceMode: rawData.sourceMode === "extract" ? "extract" : "upload",
      promptTemplate:
        typeof rawData.promptTemplate === "string" &&
        !rawData.promptTemplate.includes("{question}")
          ? rawData.promptTemplate
          : "",
      previewRows: Array.isArray(rawData.previewRows) ? rawData.previewRows : [],
      selectedPreviewKeys: Array.isArray(rawData.selectedPreviewKeys) ? rawData.selectedPreviewKeys : []
    },
    model: {
      ...base.model,
      ...rawModel,
      selectedTargets:
        Array.isArray(rawModel.selectedTargets) && rawModel.selectedTargets.length > 0
          ? rawModel.selectedTargets
          : [...defaultLoraTargets]
    },
    hyperparameters: {
      ...base.hyperparameters,
      ...rawHyperparameters,
      epochs: Number(rawHyperparameters.epochs ?? base.hyperparameters.epochs) || 1
    },
    run: {
      ...base.run,
      ...rawRun
    }
  };
}
