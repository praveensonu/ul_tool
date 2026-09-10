import { ArrowLeft, CheckCircle2, Loader2, RefreshCw, RotateCcw, WifiOff } from "lucide-react";
import { useEffect, useState } from "react";
import { checkBackend } from "../../api";
import { useProject } from "../../state/ProjectContext";
import ThemeControl from "../ui/ThemeControl";
import HelpTip, { fieldHelp } from "../ui/HelpTip";

type BackendState = "checking" | "ok" | "down";

export default function ProjectHeader({ onBack }: { onBack: () => void }) {
  const { project, renameProject, resetPipeline } = useProject();
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [backendMessage, setBackendMessage] = useState("Checking backend");

  async function refreshBackend() {
    setBackendState("checking");
    try {
      const response = await checkBackend();
      setBackendState("ok");
      setBackendMessage(response.message);
    } catch (error) {
      setBackendState("down");
      setBackendMessage(error instanceof Error ? error.message : "Backend unavailable");
    }
  }

  useEffect(() => {
    refreshBackend();
  }, []);

  return (
    <header className="project-header">
      <div className="project-header-main">
        <button className="text-button" type="button" onClick={onBack}>
          <ArrowLeft size={17} />
          Projects
        </button>
        <div className="project-title-wrap"><input className="project-title-input" aria-label="Project name" value={project.name || "Unnamed project"} onChange={(event) => renameProject(event.target.value || "Unnamed project")} /><HelpTip text={fieldHelp.projectName} /></div>
      </div>
      <div className="header-actions"><ThemeControl />{project.pendingResetFrom && <button className="reset-button" type="button" onClick={resetPipeline}><RotateCcw size={15} /><span>Reset from {project.pendingResetFrom}</span></button>}<button className={`backend-status ${backendState}`} type="button" onClick={refreshBackend}>
        {backendState === "checking" ? (
          <Loader2 className="spin" size={16} />
        ) : backendState === "ok" ? (
          <CheckCircle2 size={16} />
        ) : (
          <WifiOff size={16} />
        )}
        <span>{backendMessage}</span>
        <RefreshCw size={14} />
      </button></div>
    </header>
  );
}
