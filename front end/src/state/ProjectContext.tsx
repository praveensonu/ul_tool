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
  evaluationRequested: false,
  evaluation: null,
  evaluationMessage: null
};

type ProjectContextValue = {
  project: Project;
  renameProject: (name: string) => void;
  updateDataInput: (patch: Partial<ProjectDataConfig>) => void;
  setDataPrepared: (patch: Partial<ProjectDataConfig>) => void;
  setPreviewSelection: (keys: string[]) => void;
  setDatasetUpload: (response: DatasetUploadResponse | null, error?: string | null) => void;
  updateModel: (patch: Partial<ProjectModelConfig>) => void;
  updateHyperparameters: (patch: Partial<ProjectHyperparametersConfig>) => void;
  updateRun: (patch: Partial<ProjectRunState>) => void;
  markStageCompleted: (stage: ProjectStage, completed?: boolean) => void;
  setLastStage: (stage: ProjectStage) => void;
  saveNow: () => Promise<void>;
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
      updateDataInput: (patch) =>
        commit((current) => ({
          ...current,
          data: {
            ...current.data,
            ...patch,
            preparedForgetFile: null,
            preparedRetainFile: null,
            previewRows: [],
            selectedPreviewKeys: [],
            previewReady: false,
            previewFilterable: true,
            uploadResponse: null,
            backendUploadError: null
          },
          completedStages: {
            data: false,
            model: false,
            hyperparameters: false,
            running: false
          },
          run: emptyRun
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
            preparedForgetFile: null,
            uploadResponse: null,
            backendUploadError: null
          },
          completedStages: {
            data: false,
            model: false,
            hyperparameters: false,
            running: false
          },
          run: emptyRun
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
      updateModel: (patch) =>
        commit((current) => ({
          ...current,
          model: { ...current.model, ...patch },
          completedStages: {
            ...current.completedStages,
            model: false,
            hyperparameters: false,
            running: false
          },
          run: emptyRun
        })),
      updateHyperparameters: (patch) =>
        commit((current) => ({
          ...current,
          hyperparameters: { ...current.hyperparameters, ...patch },
          completedStages: {
            ...current.completedStages,
            hyperparameters: false,
            running: false
          },
          run: emptyRun
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
