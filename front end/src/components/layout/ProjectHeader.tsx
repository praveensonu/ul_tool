import { ArrowLeft, CheckCircle2, Loader2, RefreshCw, WifiOff } from "lucide-react";
import { useEffect, useState } from "react";
import { checkBackend } from "../../api";
import { useProject } from "../../state/ProjectContext";

type BackendState = "checking" | "ok" | "down";

export default function ProjectHeader({ onBack }: { onBack: () => void }) {
  const { project, renameProject } = useProject();
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
        <input
          className="project-title-input"
          aria-label="Project name"
          value={project.name}
          onChange={(event) => renameProject(event.target.value)}
        />
      </div>

      <button className={`backend-status ${backendState}`} type="button" onClick={refreshBackend}>
        {backendState === "checking" ? (
          <Loader2 className="spin" size={16} />
        ) : backendState === "ok" ? (
          <CheckCircle2 size={16} />
        ) : (
          <WifiOff size={16} />
        )}
        <span>{backendMessage}</span>
        <RefreshCw size={14} />
      </button>
    </header>
  );
}
