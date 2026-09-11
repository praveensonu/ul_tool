import { useEffect, useState } from "react";
import type { CSSProperties } from "react";
import { AlertTriangle, BarChart3, CheckCircle2, Gauge as GaugeIcon, Minus } from "lucide-react";
import type { EvaluationResponse } from "../../types";

const PRE_COLOR = "#62718a";
const POST_COLOR = "#267a59";

type Direction = "up" | "down" | "stable";
type Outcome = "good" | "neutral" | "bad";
type MetricRow = {
  label: string;
  pre: number;
  post: number;
  direction: Direction;
};

function formatMetric(value: number) {
  if (!Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 ? value.toExponential(2) : value.toFixed(4);
}

function outcome(pre: number, post: number, direction: Direction): Outcome {
  const tolerance = Math.max(Math.abs(pre) * 0.02, 0.002);
  const delta = post - pre;
  if (Math.abs(delta) <= tolerance) return direction === "stable" ? "good" : "neutral";
  if (direction === "stable") return "bad";
  return (direction === "up" ? delta > 0 : delta < 0) ? "good" : "bad";
}

function StatusMark({ state }: { state: Outcome }) {
  const labels = {
    good: "Desired",
    neutral: "Neutral",
    bad: "Attention",
  };

  return (
    <span
      className={`metric-status ${state}`}
      title={
        state === "good"
          ? "Desired change"
          : state === "bad"
            ? "Change needs attention"
            : "Neutral change"
      }
    >
      {state === "good" ? (
        <CheckCircle2 size={17} strokeWidth={2.4} />
      ) : state === "bad" ? (
        <AlertTriangle size={17} strokeWidth={2.4} />
      ) : (
        <Minus size={17} strokeWidth={2.5} />
      )}
      <span>{labels[state]}</span>
    </span>
  );
}

function Gauge({
  value,
  maximum = 1,
  color,
  label,
}: {
  value: number;
  maximum?: number;
  color: string;
  label: string;
}) {
  const safeMaximum = maximum > 0 ? maximum : 1;
  const normalized = Math.max(0, Math.min(1, value / safeMaximum));
  const angle = -90 + normalized * 180;

  return (
    <div className="gauge-plot">
      <svg viewBox="0 0 180 105" role="img" aria-label={`${label}: ${formatMetric(value)}`}>
        <path className="gauge-track" d="M20 90 A70 70 0 0 1 160 90" pathLength="100" />
        <path
          className="gauge-fill"
          style={{
            stroke: color,
            "--gauge-offset": String(100 - normalized * 100),
          } as CSSProperties}
          d="M20 90 A70 70 0 0 1 160 90"
          pathLength="100"
        />
        <g className="gauge-quartiles">
          <path d="M90 14v9" />
          <path d="M40 37l7 7" />
          <path d="M140 37l-7 7" />
        </g>
        <line
          className="gauge-needle"
          x1="90"
          y1="90"
          x2="90"
          y2="31"
          style={{ "--gauge-angle": `${angle}deg` } as CSSProperties}
        />
        <circle className="gauge-hub-ring" cx="90" cy="90" r="9" />
        <circle cx="90" cy="90" r="5.5" fill={color} />
      </svg>
      <strong>{formatMetric(value)}</strong>
      <small>{label}</small>
    </div>
  );
}

function GaugeComparison({
  title,
  subtitle,
  rows,
  adaptiveScale = false,
}: {
  title: string;
  subtitle: string;
  rows: MetricRow[];
  adaptiveScale?: boolean;
}) {
  return (
    <section className="evaluation-chart-card gauge-card">
      <div className="chart-heading">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>

      {rows.map((row) => {
        const state = outcome(row.pre, row.post, row.direction);
        const maximum = adaptiveScale ? Math.max(row.pre, row.post, 0.000001) * 1.12 : 1;
        return (
          <div className={`gauge-metric outcome-${state}`} key={row.label}>
            <div className="gauge-title">
              <strong>{row.label}</strong>
              <StatusMark state={state} />
            </div>
            <div className="gauge-pair">
              <Gauge value={row.pre} maximum={maximum} color={PRE_COLOR} label="Pre" />
              <Gauge value={row.post} maximum={maximum} color={POST_COLOR} label="Post" />
            </div>
          </div>
        );
      })}
    </section>
  );
}

function ComparisonBars({
  title,
  subtitle,
  rows,
}: {
  title: string;
  subtitle: string;
  rows: MetricRow[];
}) {
  const maximum = Math.max(...rows.flatMap((row) => [row.pre, row.post]), 0.000001);
  return (
    <section className="evaluation-chart-card">
      <div className="chart-heading">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>
      <div className="comparison-bars">
        {rows.map((row) => {
          const state = outcome(row.pre, row.post, row.direction);
          return (
            <div className={`comparison-row outcome-${state}`} key={row.label}>
              <div className="comparison-row-head">
                <strong>{row.label}</strong>
                <div className="comparison-row-meta">
                  <StatusMark state={state} />
                  <div className="bar-mini-legend" aria-label="Bar legend">
                    <span><i style={{ background: PRE_COLOR }} />Pre</span>
                    <span><i style={{ background: POST_COLOR }} />Post</span>
                  </div>
                </div>
              </div>
              <div className="bar-pair">
                <div className="bar-track">
                  <span className="metric-bar pre" style={{ width: `${Math.max(1, row.pre / maximum * 100)}%` }} />
                  <em>{formatMetric(row.pre)}</em>
                </div>
                <div className="bar-track">
                  <span className="metric-bar post" style={{ width: `${Math.max(1, row.post / maximum * 100)}%` }} />
                  <em>{formatMetric(row.post)}</em>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default function EvaluationDashboard({ result }: { result: EvaluationResponse }) {
  const [chartMode, setChartMode] = useState<"gauges" | "bars">(() => {
    try {
      return window.localStorage.getItem("ascent-evaluation-chart") === "bars" ? "bars" : "gauges";
    } catch {
      return "gauges";
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem("ascent-evaluation-chart", chartMode);
    } catch {
      // The preference remains active for this session when storage is unavailable.
    }
  }, [chartMode]);

  const pre = result.pre_unlearning;
  const post = result.post_unlearning;
  const overall: MetricRow[] = [
    { label: "Forget quality", pre: pre.forget_quality.score, post: post.forget_quality.score, direction: "up" },
    { label: "Model utility", pre: pre.model_utility.score, post: post.model_utility.score, direction: "up" },
  ];
  const perplexity: MetricRow[] = [
    { label: "Forget set", pre: pre.forget_quality.average_perplexity, post: post.forget_quality.average_perplexity, direction: "up" },
    { label: "Retain set", pre: pre.model_utility.average_perplexity, post: post.model_utility.average_perplexity, direction: "down" },
  ];
  const forget: MetricRow[] = [
    { label: "Conditional probability", pre: pre.forget_quality.mean_conditional_probability, post: post.forget_quality.mean_conditional_probability, direction: "down" },
    { label: "ROUGE-L", pre: pre.forget_quality.mean_rouge_l, post: post.forget_quality.mean_rouge_l, direction: "down" },
  ];
  const retain: MetricRow[] = [
    { label: "Conditional probability", pre: pre.model_utility.mean_conditional_probability, post: post.model_utility.mean_conditional_probability, direction: "up" },
    { label: "ROUGE-L", pre: pre.model_utility.mean_rouge_l, post: post.model_utility.mean_rouge_l, direction: "up" },
    { label: "Cosine similarity", pre: pre.model_utility.mean_cosine_similarity, post: post.model_utility.mean_cosine_similarity, direction: "up" },
  ];
  const benchmarks: MetricRow[] = pre.benchmarks && post.benchmarks ? [
    { label: "MMLU", pre: pre.benchmarks.mmlu, post: post.benchmarks.mmlu, direction: "stable" },
    { label: "GPQA", pre: pre.benchmarks.gpqa, post: post.benchmarks.gpqa, direction: "stable" },
  ] : [];
  const details = [...overall, ...perplexity, ...benchmarks, ...forget, ...retain];

  return (
    <div className="evaluation-dashboard">
      <div className="dashboard-title-row">
        <div>
          <span className="eyebrow">Comparison dashboard</span>
          <h2>Before and after unlearning</h2>
          <p>Green marks a desired change, amber a neutral result, and red a result requiring attention.</p>
        </div>
        <div className="dashboard-tools">
          <div className="chart-mode-toggle" role="group" aria-label="Evaluation chart style">
            <button type="button" className={chartMode === "gauges" ? "active" : ""} aria-pressed={chartMode === "gauges"} onClick={() => setChartMode("gauges")}>
              <GaugeIcon size={15} /> Gauges
            </button>
            <button type="button" className={chartMode === "bars" ? "active" : ""} aria-pressed={chartMode === "bars"} onClick={() => setChartMode("bars")}>
              <BarChart3 size={15} /> Bars
            </button>
          </div>
          <div className="embedding-pill">Embeddings: {result.embedding_model_name}</div>
        </div>
      </div>

      <section className="primary-results">
        <span className="section-priority">Primary outcomes</span>
        <div className="dashboard-chart-grid">
          {chartMode === "gauges" ? (
            <>
              <GaugeComparison title="Overall scores" subtitle="Core balance between forgetting and retained utility." rows={overall} />
              <GaugeComparison title="Perplexity" subtitle="Forget perplexity should rise; retain perplexity should stay controlled." rows={perplexity} adaptiveScale />
            </>
          ) : (
            <>
              <ComparisonBars title="Overall scores" subtitle="Core balance between forgetting and retained utility." rows={overall} />
              <ComparisonBars title="Perplexity" subtitle="Forget perplexity should rise; retain perplexity should stay controlled." rows={perplexity} />
            </>
          )}
        </div>
        {benchmarks.length > 0 && (chartMode === "gauges" ? (
          <GaugeComparison title="Benchmark accuracy" subtitle="MMLU and GPQA general capability before and after unlearning." rows={benchmarks} />
        ) : (
          <ComparisonBars title="Benchmark accuracy" subtitle="MMLU and GPQA general capability before and after unlearning." rows={benchmarks} />
        ))}
      </section>

      <section className="secondary-results">
        <span className="section-priority secondary">Diagnostic outcomes</span>
        <div className="dashboard-chart-grid">
          {chartMode === "gauges" ? (
            <>
              <GaugeComparison title="Forget-set diagnostics" subtitle="Lower post-unlearning values indicate stronger forgetting." rows={forget} />
              <GaugeComparison title="Retain-set diagnostics" subtitle="Higher post-unlearning values indicate better preservation." rows={retain} />
            </>
          ) : (
            <>
              <ComparisonBars title="Forget-set diagnostics" subtitle="Lower post-unlearning values indicate stronger forgetting." rows={forget} />
              <ComparisonBars title="Retain-set diagnostics" subtitle="Higher post-unlearning values indicate better preservation." rows={retain} />
            </>
          )}
        </div>
      </section>



      <details className="metric-table-card">
        <summary>Detailed numerical comparison</summary>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Metric</th><th>Pre</th><th>Post</th><th>Signal</th></tr></thead>
            <tbody>
              {details.map((row, index) => {
                const state = outcome(row.pre, row.post, row.direction);
                return (
                  <tr className={`outcome-${state}`} key={`${row.label}-${index}`}>
                    <th>{row.label}</th>
                    <td className="pre-value">{formatMetric(row.pre)}</td>
                    <td className="post-value">{formatMetric(row.post)}</td>
                    <td><StatusMark state={state} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
