import { HelpCircle } from "lucide-react";
import { useId, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { createPortal } from "react-dom";

export const fieldHelp = {
  projectName: "A local name used to identify this project and label backend experiments.",
  sourceMode: "Upload uses ready-made forget/retain sets. Extract derives both sets from a full dataset and a poison set.",
  gpu: "GPUs used for extraction, unlearning, and generation. Evaluation scoring uses the first selected GPU.",
  forgetSet: "Examples the model should forget. Supported formats: CSV, JSON, JSONL, and Parquet.",
  retainSet: "Optional examples whose knowledge should be preserved. Some unlearning methods require this set.",
  fullDataset: "The complete source dataset from which extraction selects forget and retain examples.",
  poisonSet: "Reference examples used to compute poison gradients during extraction.",
  extractionMethod: "RASLIK ranks gradient influence; GRACE also promotes diversity through clustering.",
  forgetSamples: "Number of highest-priority examples placed in the extracted forget set.",
  retainSamples: "Number of representative examples placed in the extracted retain set.",
  candidatePool: "Number of top-ranked candidates GRACE considers before diversity clustering.",
  retainClusters: "Number of clusters used to distribute retained examples across the dataset.",
  gradientBatch: "Examples processed per GPU while calculating gradients. Lower this if GPU memory is limited.",
  model: "A Hugging Face repository identifier or a local model path.",
  hfToken: "Optional access token for gated or private Hugging Face models. It remains in this browser project.",
  maxLength: "Maximum token length used while extracting gradients.",
  adapter: "Optional path to an existing adapter or LoRA checkpoint.",
  keepGradients: "Keeps cached training and poison gradients after extraction for reuse or inspection.",
  prompt: "Include {question} where the question column should appear. The answer is appended after the complete template. No chat template is added automatically.",
  method: "Choose full-model unlearning, train a new LoRA adapter, or continue from an existing adapter.",
  loraTargets: "Transformer modules updated by LoRA. The backend advertises additional supported modules when available.",
  unlearningMethod: "The optimization objective used to remove targeted knowledge.",
  trainingLength: "Stop after an exact number of optimizer steps or after complete passes through the dataset.",
  learningRate: "The optimizer step size. Larger values create stronger but potentially less stable updates.",
  contextLength: "Maximum token sequence length processed by the model.",
  batchSize: "Examples processed in each forward/backward pass.",
  gradAccum: "Number of batches accumulated before an optimizer update; increases effective batch size.",
  saveSteps: "How frequently the unlearning process writes a checkpoint.",
  weightDecay: "Regularization applied to model weights during optimization.",
  benchmarks: "Also evaluates general knowledge with MMLU and GPQA; this adds runtime.",
  evaluationBatch: "Examples generated and scored together. Reduce it if GPU memory is limited.",
  embeddingModel: "Sentence-transformers model used for semantic similarity scoring after language models are unloaded.",
  maxTokens: "Maximum number of new tokens generated for each evaluated answer."
} as const;

export default function HelpTip({ text }: { text: string }) {
  const anchorRef = useRef<HTMLSpanElement>(null);
  const tooltipId = useId();
  const [popoverStyle, setPopoverStyle] = useState<CSSProperties | null>(null);

  const showPopover = () => {
    const anchor = anchorRef.current;
    if (!anchor) return;

    const rect = anchor.getBoundingClientRect();
    const tooltipWidth = Math.min(280, window.innerWidth - 24);
    const halfWidth = tooltipWidth / 2;
    const left = Math.max(halfWidth + 12, Math.min(window.innerWidth - halfWidth - 12, rect.left + rect.width / 2));
    const showBelow = rect.top < 125;

    setPopoverStyle({
      left,
      top: showBelow ? rect.bottom + 10 : rect.top - 10,
      width: tooltipWidth,
      transform: showBelow ? "translate(-50%, 0)" : "translate(-50%, -100%)",
    });
  };

  return (
    <span
      ref={anchorRef}
      className="help-tip"
      tabIndex={0}
      aria-label={`Help: ${text}`}
      aria-describedby={popoverStyle ? tooltipId : undefined}
      onMouseEnter={showPopover}
      onMouseLeave={() => setPopoverStyle(null)}
      onFocus={showPopover}
      onBlur={() => setPopoverStyle(null)}
    >
      <HelpCircle size={15} aria-hidden="true" />
      {popoverStyle && createPortal(
        <span id={tooltipId} className="help-popover help-popover-portal" role="tooltip" style={popoverStyle}>
          {text}
        </span>,
        document.body,
      )}
    </span>
  );
}

export function FieldLabel({ children, help }: { children: ReactNode; help: string }) {
  return <span className="field-label">{children}<HelpTip text={help} /></span>;
}
