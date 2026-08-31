import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import ProjectHeader from "../components/layout/ProjectHeader";
import StepNavigation from "../components/project/StepNavigation";
import DataStage from "../components/stages/DataStage";
import HyperparametersStage from "../components/stages/HyperparametersStage";
import ModelStage from "../components/stages/ModelStage";
import RunningStage from "../components/stages/RunningStage";
import EvaluationStage from "../components/stages/EvaluationStage";
import { ProjectProvider, useProject } from "../state/ProjectContext";
import { getProject } from "../storage/projectStorage";
import type { Project, ProjectStage } from "../types";
import {
  canAccessStage,
  getNextStage,
  getPreviousStage,
  isProjectStage,
  isStageValid
} from "../utils/projectValidation";

function StageContent({ stage }: { stage: ProjectStage }) {
  if (stage === "data") return <DataStage />;
  if (stage === "model") return <ModelStage />;
  if (stage === "hyperparameters") return <HyperparametersStage />;
  if (stage === "running") return <RunningStage />;
  return <EvaluationStage />;
}

function ProjectWorkspace({ requestedStage }: { requestedStage?: string }) {
  const navigate = useNavigate();
  const { project, setLastStage, saveNow, markStageCompleted } = useProject();

  const requestedIsValid = isProjectStage(requestedStage);
  const preferredStage = requestedIsValid ? requestedStage : project.lastStage;
  const stage = canAccessStage(project, preferredStage) ? preferredStage : "data";

  useEffect(() => {
    if (!requestedIsValid || requestedStage !== stage) {
      navigate(`/project/${project.id}/${stage}`, { replace: true });
    }
  }, [navigate, project.id, requestedIsValid, requestedStage, stage]);

  useEffect(() => {
    if (project.lastStage !== stage) setLastStage(stage);
  }, [project.lastStage, setLastStage, stage]);

  async function goHome() {
    await saveNow();
    navigate("/");
  }

  function goToStage(nextStage: ProjectStage) {
    if (!canAccessStage(project, nextStage)) return;
    setLastStage(nextStage);
    navigate(`/project/${project.id}/${nextStage}`);
  }

  function continueToNext() {
    const next = getNextStage(stage);
    if (!next || !isStageValid(project, stage)) return;
    markStageCompleted(stage, true);
    setLastStage(next);
    navigate(`/project/${project.id}/${next}`);
  }

  const previous = getPreviousStage(stage);
  const next = getNextStage(stage);
  const canContinue = next ? isStageValid(project, stage) : false;

  return (
    <main className="project-shell">
      <ProjectHeader onBack={goHome} />
      <StepNavigation project={project} currentStage={stage} onSelect={goToStage} />

      <div className="stage-container">
        <StageContent stage={stage} />

        <footer className="stage-footer">
          <button
            className="secondary-button"
            type="button"
            disabled={!previous}
            onClick={() => previous && goToStage(previous)}
          >
            Previous
          </button>

          {next && (
            <button
              className="primary-button"
              type="button"
              disabled={!canContinue}
              onClick={continueToNext}
            >
              Continue to {next === "hyperparameters" ? "hyperparameters" : next}
            </button>
          )}
        </footer>
      </div>
    </main>
  );
}

export default function ProjectPage() {
  const { projectId, stage } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      if (!projectId) {
        navigate("/", { replace: true });
        return;
      }

      const stored = await getProject(projectId);
      if (cancelled) return;
      if (!stored) {
        navigate("/", { replace: true });
        return;
      }

      setProject(stored);
      setLoading(false);
    }

    load().catch(() => navigate("/", { replace: true }));
    return () => {
      cancelled = true;
    };
  }, [navigate, projectId]);

  if (loading || !project) {
    return <main className="loading-screen">Loading project...</main>;
  }

  return (
    <ProjectProvider initialProject={project}>
      <ProjectWorkspace requestedStage={stage} />
    </ProjectProvider>
  );
}
