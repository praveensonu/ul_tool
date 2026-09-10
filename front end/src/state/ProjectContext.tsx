import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";
import { saveProject } from "../storage/projectStorage";
import type {
  DatasetUploadResponse,
  DatasetExtractionResponse,
  ExtractionJobStatus,
  Project,
  ProjectDataConfig,
  ProjectHyperparametersConfig,
  ProjectModelConfig,
  ProjectRunState,
  ProjectStage
} from "../types";

const emptyRun: ProjectRunState = {
  config: null,
  training: null,
  message: null,
  embeddingModelName: "",
  evaluationMaxNewTokens: 256,
  evaluationBatchSize: 4,
  includeBenchmarks: false,
  evaluationJob: null,
  evaluation: null,
  evaluationMessage: null
};

const stageRank: ProjectStage[] = ["gpu", "data", "model", "hyperparameters", "running", "evaluation"];
function earliestStage(current: ProjectStage | null, next: ProjectStage) {
  return !current || stageRank.indexOf(next) < stageRank.indexOf(current) ? next : current;
}

type ProjectContextValue = {
  project: Project;
  renameProject: (name: string) => void;
  updateDataInput: (patch: Partial<ProjectDataConfig>) => void;
  setDataPrepared: (patch: Partial<ProjectDataConfig>) => void;
  setPreviewSelection: (keys: string[]) => void;
  setDatasetUpload: (response: DatasetUploadResponse | null, error?: string | null) => void;
  setExtractionResult: (response: DatasetExtractionResponse | null) => void;
  setExtractionJob: (job: ExtractionJobStatus | null) => void;
  updateModel: (patch: Partial<ProjectModelConfig>) => void;
  updateHyperparameters: (patch: Partial<ProjectHyperparametersConfig>) => void;
  updateRun: (patch: Partial<ProjectRunState>) => void;
  markStageCompleted: (stage: ProjectStage, completed?: boolean) => void;
  setLastStage: (stage: ProjectStage) => void;
  saveNow: () => Promise<void>;
  finishSetup: (sourceMode: ProjectDataConfig["sourceMode"]) => void;
  resetPipeline: () => void;
};

const ProjectContext = createContext<ProjectContextValue | null>(null);

export function ProjectProvider({
  initialProject,
  children
}: {
  initialProject: Project;
  children: ReactNode;
}) {
  const [project, setProject] = useState(initialProject);

  const commit = useCallback((updater: (current: Project) => Project) => {
    setProject((current) => ({
      ...updater(current),
      updatedAt: new Date().toISOString()
    }));
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      saveProject(project).catch((error) => {
        console.error("Could not auto-save project", error);
      });
    }, 250);

    return () => window.clearTimeout(timer);
  }, [project]);

  const value = useMemo<ProjectContextValue>(
    () => ({
      project,
      renameProject: (name) => commit((current) => ({ ...current, name })),
      finishSetup: (sourceMode) => commit((current) => ({
        ...current,
        setupComplete: true,
        data: { ...current.data, sourceMode }
      })),
      updateDataInput: (patch) =>
        commit((current) => ({
          ...current,
          data: {
            ...current.data,
            ...patch
          },
          pendingResetFrom: current.data.previewReady || current.data.uploadResponse || current.completedStages.data
            ? earliestStage(current.pendingResetFrom, "data") : current.pendingResetFrom
        })),
      setDataPrepared: (patch) =>
        commit((current) => ({
          ...current,
          data: {
            ...current.data,
            ...patch
          }
        })),
      setPreviewSelection: (keys) =>
        commit((current) => ({
          ...current,
          data: {
            ...current.data,
            selectedPreviewKeys: keys,
            ...(current.completedStages.data ? {} : { preparedForgetFile: null, uploadResponse: null, backendUploadError: null })
          },
          pendingResetFrom: current.completedStages.data ? earliestStage(current.pendingResetFrom, "data") : current.pendingResetFrom
        })),
      setDatasetUpload: (response, error = null) =>
        commit((current) => ({
          ...current,
          data: {
            ...current.data,
            uploadResponse: response,
            backendUploadError: error
          }
        })),
      setExtractionResult: (response) =>
        commit((current) => ({
          ...current,
          data: {
            ...current.data,
            extractionResponse: response,
            uploadResponse: response,
            backendUploadError: null
          }
        })),
      setExtractionJob: (job) =>
        commit((current) => ({
          ...current,
          data: {
            ...current.data,
            extractionJob: job
          }
        })),
      updateModel: (patch) =>
        commit((current) => ({
          ...current,
          model: { ...current.model, ...patch },
          pendingResetFrom: current.completedStages.model || current.completedStages.running || (Boolean(patch.gpuIds) && current.completedStages.data)
            ? earliestStage(current.pendingResetFrom, patch.gpuIds ? "gpu" : (patch.modelName && current.completedStages.data ? "data" : "model"))
            : current.pendingResetFrom
        })),
      updateHyperparameters: (patch) =>
        commit((current) => ({
          ...current,
          hyperparameters: { ...current.hyperparameters, ...patch },
          pendingResetFrom: current.completedStages.hyperparameters || current.completedStages.running
            ? earliestStage(current.pendingResetFrom, "hyperparameters")
            : current.pendingResetFrom
        })),
      updateRun: (patch) =>
        commit((current) => ({
          ...current,
          run: { ...current.run, ...patch }
        })),
      markStageCompleted: (stage, completed = true) =>
        commit((current) => ({
          ...current,
          completedStages: {
            ...current.completedStages,
            [stage]: completed
          }
        })),
      setLastStage: (stage) =>
        commit((current) => ({
          ...current,
          lastStage: stage
        })),
      resetPipeline: () => commit((current) => {
        const from = current.pendingResetFrom;
        if (!from) return current;
        const order: ProjectStage[] = ["gpu", "data", "model", "hyperparameters", "running", "evaluation"];
        const start = order.indexOf(from);
        const completedStages = { ...current.completedStages };
        order.slice(start).forEach((stage) => { completedStages[stage] = false; });
        const resetData = start <= order.indexOf("data");
        return {
          ...current,
          pendingResetFrom: null,
          completedStages,
          lastStage: from,
          data: resetData ? {
            ...current.data,
            preparedForgetFile: null,
            preparedRetainFile: null,
            previewRows: [],
            selectedPreviewKeys: [],
            previewReady: false,
            previewFilterable: true,
            uploadResponse: null,
            extractionJob: null,
            extractionResponse: null,
            backendUploadError: null
          } : current.data,
          run: start <= order.indexOf("running") ? emptyRun : current.run
        };
      }),
      saveNow: () => saveProject(project)
    }),
    [commit, project]
  );

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>;
}

export function useProject() {
  const context = useContext(ProjectContext);
  if (!context) throw new Error("useProject must be used inside ProjectProvider.");
  return context;
}
