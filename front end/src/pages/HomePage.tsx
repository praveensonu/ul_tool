import { Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import ProjectCard from "../components/project/ProjectCard";
import { createDefaultProject } from "../defaults";
import { deleteProject, listProjects, saveProject } from "../storage/projectStorage";
import type { Project } from "../types";

export default function HomePage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      setProjects(await listProjects());
    } catch (storageError) {
      setError(storageError instanceof Error ? storageError.message : "Could not load projects.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleCreate() {
    const project = createDefaultProject();
    await saveProject(project);
    navigate(`/project/${project.id}/data`);
  }

  async function handleDelete(project: Project) {
    if (!window.confirm(`Delete ${project.name || "this project"}?`)) return;
    await deleteProject(project.id);
    await refresh();
  }

  return (
    <main className="home-shell">
      <header className="home-header">
        <div>
          <p className="eyebrow">Ascent unlearning</p>
          <h1>Projects</h1>
          <p className="muted">Create a new unlearning project or continue from your previous history.</p>
        </div>
        <button className="primary-button" type="button" onClick={handleCreate}>
          <Plus size={18} />
          New unlearning project
        </button>
      </header>

      {error && <div className="notice error">{error}</div>}

      {loading ? (
        <p className="muted">Loading projects...</p>
      ) : projects.length === 0 ? (
        <section className="empty-state">
          <h2>No projects yet</h2>
          <p>Create your first project. Its data, settings, and last open stage are saved in this browser.</p>
          <button className="primary-button" type="button" onClick={handleCreate}>
            <Plus size={18} />
            New unlearning project
          </button>
        </section>
      ) : (
        <section className="project-grid">
          {projects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onOpen={() => navigate(`/project/${project.id}/${project.lastStage}`)}
              onDelete={() => handleDelete(project)}
            />
          ))}
        </section>
      )}
    </main>
  );
}
