import { retainRequiredMethods } from "../defaults";
import type { Project, ProjectStage } from "../types";

const stageOrder: ProjectStage[] = ["gpu", "data", "model", "hyperparameters", "running", "evaluation"];

export function isDataValid(project: Project) {
  if (project.data.sourceMode === "extract") {
    return Boolean(
      project.data.fullFile &&
        project.data.poisonFile &&
        project.data.promptTemplate.trim().length > 0 &&
        project.data.extractionResponse &&
        project.data.uploadResponse?.forget_set_path &&
        project.data.uploadResponse?.retain_set_path
    );
  }

  const sourceOk =
    project.data.sourceMode === "upload" && Boolean(project.data.forgetFile);

  return Boolean(
    sourceOk &&
      project.data.promptTemplate.trim().length > 0 &&
      project.data.previewReady &&
      project.data.preparedForgetFile &&
      project.data.selectedPreviewKeys.length > 0
  );
}

export function isModelValid(project: Project) {
  const modelNameOk = project.model.modelName.trim().length > 0;
  const adaptorOk =
    project.model.method !== "adaptor" || project.model.adaptorPath.trim().length > 0;
  const loraOk =
    project.model.method !== "lora" || project.model.selectedTargets.length > 0;

  return modelNameOk && adaptorOk && loraOk;
}

export function isHyperparametersValid(project: Project) {
  const h = project.hyperparameters;
  const scheduleOk = h.stepMode === "max_steps" ? h.maxSteps > 0 : h.epochs > 0;
  const lr = Number(h.learningRate);
  const learningRateOk = Number.isFinite(lr) && lr > 0;
  const contextOk = h.contextLength > 0 && (h.contextLength & (h.contextLength - 1)) === 0;
  const retainOk =
    !retainRequiredMethods.includes(h.unlearningMethod) ||
    Boolean(
      project.data.uploadResponse?.retain_set_path ??
        project.data.preparedRetainFile ??
        project.data.retainFile
    );

  return (
    scheduleOk &&
    retainOk &&
    learningRateOk &&
    contextOk &&
    h.batchSize > 0 &&
    h.gradAccum > 0 &&
    h.saveSteps > 0 &&
    h.weightDecay >= 0
  );
}

export function isStageValid(project: Project, stage: ProjectStage) {
  if (stage === "gpu") return project.completedStages.gpu && project.model.gpuIds.length > 0;
  if (stage === "data") return isDataValid(project);
  if (stage === "model") return isModelValid(project);
  if (stage === "hyperparameters") return isHyperparametersValid(project);
  if (stage === "running") return project.run.training?.status === "success";
  return project.run.evaluation?.status === "success";
}

export function isStageComplete(project: Project, stage: ProjectStage) {
  return project.completedStages[stage];
}

export function canAccessStage(project: Project, stage: ProjectStage) {
  if (stage === "gpu") return true;
  if (!project.completedStages.gpu || !project.model.gpuIds.length) return false;
  if (stage === "data") return true;
  if (stage === "model") return project.completedStages.data;
  if (stage === "hyperparameters") {
    return project.completedStages.data && project.completedStages.model;
  }
  if (stage === "running") return (
    project.completedStages.data &&
    project.completedStages.model &&
    project.completedStages.hyperparameters
  );
  return (
    project.completedStages.running &&
    project.run.training?.status === "success"
  );
}

export function getPreviousStage(stage: ProjectStage): ProjectStage | null {
  const index = stageOrder.indexOf(stage);
  return index > 0 ? stageOrder[index - 1] : null;
}

export function getNextStage(stage: ProjectStage): ProjectStage | null {
  const index = stageOrder.indexOf(stage);
  return index >= 0 && index < stageOrder.length - 1 ? stageOrder[index + 1] : null;
}

export function isProjectStage(value: string | undefined): value is ProjectStage {
  return value !== undefined && stageOrder.includes(value as ProjectStage);
}
