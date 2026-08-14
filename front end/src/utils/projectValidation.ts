import type { Project, ProjectStage } from "../types";

const stageOrder: ProjectStage[] = ["data", "model", "hyperparameters", "running"];

export function isDataValid(project: Project) {
  const sourceOk =
    project.data.sourceMode === "upload"
      ? Boolean(project.data.forgetFile)
      : Boolean(project.data.fullFile && project.data.poisonFile);

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

  return (
    scheduleOk &&
    learningRateOk &&
    contextOk &&
    h.batchSize > 0 &&
    h.gradAccum > 0 &&
    h.saveSteps > 0 &&
    h.weightDecay >= 0
  );
}

export function isStageValid(project: Project, stage: ProjectStage) {
  if (stage === "data") return isDataValid(project);
  if (stage === "model") return isModelValid(project);
  if (stage === "hyperparameters") return isHyperparametersValid(project);
  return true;
}

export function isStageComplete(project: Project, stage: ProjectStage) {
  return project.completedStages[stage];
}

export function canAccessStage(project: Project, stage: ProjectStage) {
  if (stage === "data") return true;
  if (stage === "model") return project.completedStages.data;
  if (stage === "hyperparameters") {
    return project.completedStages.data && project.completedStages.model;
  }
  return (
    project.completedStages.data &&
    project.completedStages.model &&
    project.completedStages.hyperparameters
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
