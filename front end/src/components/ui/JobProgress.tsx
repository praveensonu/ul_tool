import { Check, Loader2 } from "lucide-react";
import type { ProgressEvent } from "../../types";

function phaseFor(stage: string) {
  const value = stage.toLowerCase();
  if (value.includes("post") || value.includes("unlearned")) return "post";
  return "pre";
}

function progressPercent(message?: string) {
  if (!message) return null;
  const rowRange = message.match(/rows?\s+(\d+)\s*[-–—]\s*(\d+)\s+of\s+(\d+)/i);
  if (rowRange) return Math.min(100, Math.round((Number(rowRange[2]) / Number(rowRange[3])) * 100));

  const completedRows = message.match(/(?:rows?|items?|samples?)\s+(\d+)\s+(?:of|\/)\s*(\d+)/i);
  if (completedRows) return Math.min(100, Math.round((Number(completedRows[1]) / Number(completedRows[2])) * 100));

  const batches = message.match(/batch\s+(\d+)\s+(?:of|\/)\s*(\d+)/i);
  if (batches) return Math.min(100, Math.round((Number(batches[1]) / Number(batches[2])) * 100));
  return null;
}

export default function JobProgress({ title, status, message, progress, splitEvaluation = false }: { title: string; status: string; message: string; progress: ProgressEvent[]; splitEvaluation?: boolean }) {
  const active = ["queued", "running", "cancelling"].includes(status);
  const latest = progress[progress.length - 1];
  const groups = splitEvaluation
    ? [{ key: "pre", label: "Pre-unlearning", events: progress.filter((event) => phaseFor(event.stage) === "pre") }, { key: "post", label: "Post-unlearning", events: progress.filter((event) => phaseFor(event.stage) === "post") }]
    : [{ key: "process", label: title, events: progress }];

  return (
    <section className="game-loader" aria-live="polite">
      <div className="loader-orbit"><span /><span /><span /><div>{active ? <Loader2 className="spin" size={24} /> : <Check size={24} />}</div></div>
      <div className="loader-content">
        <div className="loader-heading"><div><span className="eyebrow">{title}</span><strong>{message}</strong></div><span className={`job-status ${status}`}>{status}</span></div>
        <div className={`macro-progress ${splitEvaluation ? "split" : ""}`}>
          {groups.map((group) => {
            const groupLatest = group.events[group.events.length - 1];
            const done = group.events.length > 0 && (!active || latest !== groupLatest);
            const current = latest && group.events.includes(latest);
            const percent = done ? 100 : progressPercent(groupLatest?.message);
            return <div className={`macro-phase ${done ? "done" : ""} ${current ? "current" : ""}`} key={group.key}>
              <span>{done ? <Check size={15} /> : current ? <Loader2 className="spin" size={15} /> : null}</span>
              <div>
                <div className="phase-title"><strong>{group.label}</strong>{percent !== null && <b>{percent}%</b>}</div>
                <small>{groupLatest?.message ?? "Waiting"}</small>
                {percent !== null && (
                  <div className="phase-progress" role="progressbar" aria-label={`${group.label} progress`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
                    <i style={{ width: `${percent}%` }} />
                  </div>
                )}
              </div>
            </div>;
          })}
        </div>
        {latest && <p className="loader-current"><span className="pulse-dot" />{latest.stage.split("_").join(" ")}</p>}
      </div>
    </section>
  );
}
