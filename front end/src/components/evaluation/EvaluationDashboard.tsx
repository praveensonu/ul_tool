import type { EvaluationResponse } from "../../types";

const PRE_COLOR = "#4f6fb3";
const POST_COLOR = "#2f8a63";

function formatMetric(value: number) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toExponential(2);
  return value.toFixed(4);
}

function ComparisonBars({
  title,
  subtitle,
  rows,
  fixedMaximum
}: {
  title: string;
  subtitle: string;
  rows: Array<{ label: string; pre: number; post: number }>;
  fixedMaximum?: number;
}) {
  const observedMaximum = Math.max(
    ...rows.flatMap((row) => [row.pre, row.post]),
    0.000001
  );
  const maximum = fixedMaximum ?? observedMaximum;
  const width = (value: number) => `${Math.max(0, Math.min(100, (value / maximum) * 100))}%`;

  return (
    <section className="evaluation-chart-card">
      <div className="chart-heading">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <div className="chart-legend" aria-label="Chart legend">
          <span><i style={{ background: PRE_COLOR }} />Pre-unlearning</span>
          <span><i style={{ background: POST_COLOR }} />Post-unlearning</span>
        </div>
      </div>

      <div className="comparison-bars">
        {rows.map((row) => (
          <div className="comparison-row" key={row.label}>
            <strong>{row.label}</strong>
            <div className="bar-pair">
              <div className="bar-track">
                <span className="metric-bar pre" style={{ width: width(row.pre) }} />
                <em>{formatMetric(row.pre)}</em>
              </div>
              <div className="bar-track">
                <span className="metric-bar post" style={{ width: width(row.post) }} />
                <em>{formatMetric(row.post)}</em>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function EvaluationDashboard({ result }: { result: EvaluationResponse }) {
  const pre = result.pre_unlearning;
  const post = result.post_unlearning;
  const detailRows = [
    ["Forget quality", pre.forget_quality.score, post.forget_quality.score, "Higher is better"],
    ["Model utility", pre.model_utility.score, post.model_utility.score, "Higher is better"],
    ["Forget perplexity", pre.forget_quality.average_perplexity, post.forget_quality.average_perplexity, "Post should increase"],
    ["Retain perplexity", pre.model_utility.average_perplexity, post.model_utility.average_perplexity, "Lower is better"],
    ["Forget conditional probability", pre.forget_quality.mean_conditional_probability, post.forget_quality.mean_conditional_probability, "Post should decrease"],
    ["Retain conditional probability", pre.model_utility.mean_conditional_probability, post.model_utility.mean_conditional_probability, "Higher is better"],
    ["Forget ROUGE-L", pre.forget_quality.mean_rouge_l, post.forget_quality.mean_rouge_l, "Post should decrease"],
    ["Retain ROUGE-L", pre.model_utility.mean_rouge_l, post.model_utility.mean_rouge_l, "Higher is better"],
    ["Retain cosine similarity", pre.model_utility.mean_cosine_similarity, post.model_utility.mean_cosine_similarity, "Higher is better"]
  ] as const;

  return (
    <div className="evaluation-dashboard">
      <div className="dashboard-title-row">
        <div>
          <span className="eyebrow">Comparison dashboard</span>
          <h2>Before and after unlearning</h2>
          <p>Blue represents the original model; green represents the unlearnt model.</p>
        </div>
        <div className="embedding-pill">Embeddings: {result.embedding_model_name}</div>
      </div>

      <section className="score-card-grid">
        <article className="score-card forget">
          <span>Post-unlearning forget quality</span>
          <strong>{formatMetric(post.forget_quality.score)}</strong>
          <small>Pre-unlearning {formatMetric(pre.forget_quality.score)}</small>
        </article>
        <article className="score-card utility">
          <span>Post-unlearning model utility</span>
          <strong>{formatMetric(post.model_utility.score)}</strong>
          <small>Pre-unlearning {formatMetric(pre.model_utility.score)}</small>
        </article>
        <article className="score-card neutral">
          <span>Forget-set perplexity</span>
          <strong>{formatMetric(post.forget_quality.average_perplexity)}</strong>
          <small>Pre-unlearning {formatMetric(pre.forget_quality.average_perplexity)}</small>
        </article>
        <article className="score-card neutral">
          <span>Retain-set perplexity</span>
          <strong>{formatMetric(post.model_utility.average_perplexity)}</strong>
          <small>Pre-unlearning {formatMetric(pre.model_utility.average_perplexity)}</small>
        </article>
      </section>

      <div className="dashboard-chart-grid">
        <ComparisonBars
          title="Overall scores"
          subtitle="Harmonic means on a 0–1 scale."
          fixedMaximum={1}
          rows={[
            { label: "Forget quality", pre: pre.forget_quality.score, post: post.forget_quality.score },
            { label: "Model utility", pre: pre.model_utility.score, post: post.model_utility.score }
          ]}
        />
        <ComparisonBars
          title="Perplexity by dataset"
          subtitle="Forget perplexity should rise while retain perplexity should remain controlled."
          rows={[
            { label: "Forget set", pre: pre.forget_quality.average_perplexity, post: post.forget_quality.average_perplexity },
            { label: "Retain set", pre: pre.model_utility.average_perplexity, post: post.model_utility.average_perplexity }
          ]}
        />
        <ComparisonBars
          title="Forget-set metrics"
          subtitle="Lower probability and ROUGE-L after unlearning indicate stronger forgetting."
          fixedMaximum={1}
          rows={[
            { label: "Conditional probability", pre: pre.forget_quality.mean_conditional_probability, post: post.forget_quality.mean_conditional_probability },
            { label: "ROUGE-L", pre: pre.forget_quality.mean_rouge_l, post: post.forget_quality.mean_rouge_l }
          ]}
        />
        <ComparisonBars
          title="Retain-set metrics"
          subtitle="Higher values indicate better preservation of retained knowledge."
          fixedMaximum={1}
          rows={[
            { label: "Conditional probability", pre: pre.model_utility.mean_conditional_probability, post: post.model_utility.mean_conditional_probability },
            { label: "ROUGE-L", pre: pre.model_utility.mean_rouge_l, post: post.model_utility.mean_rouge_l },
            { label: "Cosine similarity", pre: pre.model_utility.mean_cosine_similarity, post: post.model_utility.mean_cosine_similarity }
          ]}
        />
      </div>

      {pre.benchmarks && post.benchmarks && (
        <ComparisonBars
          title="Benchmark accuracy"
          subtitle="Global MMLU (5-shot) and GPQA Main (0-shot) accuracy on a 0–1 scale."
          fixedMaximum={1}
          rows={[
            { label: "MMLU", pre: pre.benchmarks.mmlu, post: post.benchmarks.mmlu },
            { label: "GPQA", pre: pre.benchmarks.gpqa, post: post.benchmarks.gpqa }
          ]}
        />
      )}
      <section className="metric-table-card">
        <h3>Detailed comparison</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Metric</th><th>Pre-unlearning</th><th>Post-unlearning</th><th>Interpretation</th></tr></thead>
            <tbody>
              {detailRows.map(([label, preValue, postValue, interpretation]) => (
                <tr key={label}>
                  <th>{label}</th>
                  <td className="pre-value">{formatMetric(preValue)}</td>
                  <td className="post-value">{formatMetric(postValue)}</td>
                  <td>{interpretation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
