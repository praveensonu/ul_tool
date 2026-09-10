import { useEffect, useState } from "react";
import { listGpus, type GpuInfo } from "../../api";
import { useProject } from "../../state/ProjectContext";
import HelpTip, { fieldHelp } from "../ui/HelpTip";

export default function GpuStage() {
  const { project, updateModel, markStageCompleted } = useProject();
  const [gpus, setGpus] = useState<GpuInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const selected = project.model.gpuIds;
  const locked = project.run.message === "Unlearning is running." || ["queued", "running", "cancelling"].includes(project.data.extractionJob?.status ?? "") ||
    ["queued", "running", "cancelling"].includes(project.run.evaluationJob?.status ?? "");

  async function refresh() {
    setLoading(true);
    setError(null);
    markStageCompleted("gpu", false);
    try {
      const result = await listGpus();
      setGpus(result.gpus);
      const valid = selected.filter((id) => result.gpus.some((gpu) => gpu.id === id));
      if (!locked && valid.length !== selected.length) updateModel({ gpuIds: valid });
      markStageCompleted("gpu", valid.length > 0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load GPUs.");
      if (!locked) updateModel({ gpuIds: [] });
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void refresh(); }, []);

  function toggle(id: number) {
    const gpuIds = selected.includes(id) ? selected.filter((value) => value !== id) : [...selected, id];
    updateModel({ gpuIds });
    markStageCompleted("gpu", gpuIds.length > 0);
  }

  return (
    <section>
      <div className="stage-heading">
        <span className="stage-kicker">Stage 1</span>
        <div className="title-with-help"><h1>Select GPUs</h1><HelpTip text={fieldHelp.gpu}/></div>
        <p>Select one or more GPUs for data extraction, unlearning, and generation.
          All listed GPUs can be selected regardless of current usage. Evaluation scoring uses the first selected GPU.</p>
      </div>
      <button className="secondary-button" disabled={loading || locked} onClick={() => void refresh()}>
        {loading ? "Loading GPUs…" : "Refresh GPUs"}
      </button>
      {error && <p role="alert">{error}</p>}
      {!loading && !error && <p>{gpus.length} GPUs detected · {gpus.filter((gpu) => gpu.is_available).length} idle</p>}
      {gpus.map((gpu) => (
        <label key={gpu.id} className="gpu-option">
          <input type="checkbox" checked={selected.includes(gpu.id)}
            disabled={loading || locked} onChange={() => toggle(gpu.id)} />
          <span><strong>GPU {gpu.id}: {gpu.name}</strong><br />
            {gpu.is_available ? "Idle" : "In use"} · {(gpu.memory_free_mb / 1024).toFixed(1)} / {(gpu.memory_total_mb / 1024).toFixed(1)} GiB free · {gpu.utilization_percent}% utilization
          </span>
        </label>
      ))}
      <p aria-live="polite">Selected GPUs: {selected.length ? selected.join(", ") : "none"}</p>
      {locked && <p>GPU selection is locked while a job is active.</p>}
    </section>
  );
}
