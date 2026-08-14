import { useProject } from "../../state/ProjectContext";
import type { StepMode } from "../../types";

const contextOptions = [512, 1024, 2048, 4096, 8192];

export default function HyperparametersStage() {
  const { project, updateHyperparameters } = useProject();
  const h = project.hyperparameters;

  function changeMode(mode: StepMode) {
    if (mode === "epochs") {
      updateHyperparameters({ stepMode: "epochs", epochs: h.epochs > 0 ? h.epochs : 1 });
    } else {
      updateHyperparameters({ stepMode: "max_steps", maxSteps: h.maxSteps > 0 ? h.maxSteps : 100 });
    }
  }

  return (
    <section className="stage-panel">
      <div className="stage-heading">
        <span className="stage-kicker">Stage 3</span>
        <h1>Hyperparameters</h1>
        <p>Choose either max steps or epochs, then configure the remaining training parameters.</p>
      </div>

      <div className="stage-form">
        <div className="field">
          <span>Training length</span>
          <div className="segmented compact" role="group" aria-label="Training length mode">
            {(["max_steps", "epochs"] as StepMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className={h.stepMode === mode ? "active" : ""}
                onClick={() => changeMode(mode)}
              >
                {mode === "max_steps" ? "Max steps" : "Epochs"}
              </button>
            ))}
          </div>
          <small>
            Only one is sent to the orchestrator: the other value is always <code>null</code>.
          </small>
        </div>

        <div className="field-grid three">
          {h.stepMode === "max_steps" ? (
            <label className="field">
              <span>Max steps</span>
              <input
                type="number"
                min={1}
                value={h.maxSteps}
                onChange={(event) => updateHyperparameters({ maxSteps: Number(event.target.value) })}
              />
            </label>
          ) : (
            <label className="field">
              <span>Epochs</span>
              <input
                type="number"
                min={1}
                value={h.epochs}
                onChange={(event) => updateHyperparameters({ epochs: Number(event.target.value) })}
              />
            </label>
          )}

          <label className="field">
            <span>Learning rate</span>
            <input
              value={h.learningRate}
              onChange={(event) => updateHyperparameters({ learningRate: event.target.value })}
            />
          </label>

          <label className="field">
            <span>Context length</span>
            <select
              value={h.contextLength}
              onChange={(event) => updateHyperparameters({ contextLength: Number(event.target.value) })}
            >
              {contextOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Batch size</span>
            <input
              type="number"
              min={1}
              value={h.batchSize}
              onChange={(event) => updateHyperparameters({ batchSize: Number(event.target.value) })}
            />
          </label>

          <label className="field">
            <span>Gradient accumulation</span>
            <input
              type="number"
              min={1}
              value={h.gradAccum}
              onChange={(event) => updateHyperparameters({ gradAccum: Number(event.target.value) })}
            />
          </label>

          <label className="field">
            <span>Save steps</span>
            <input
              type="number"
              min={1}
              value={h.saveSteps}
              onChange={(event) => updateHyperparameters({ saveSteps: Number(event.target.value) })}
            />
          </label>

          <label className="field">
            <span>Weight decay</span>
            <input
              type="number"
              min={0}
              step="0.001"
              value={h.weightDecay}
              onChange={(event) => updateHyperparameters({ weightDecay: Number(event.target.value) })}
            />
          </label>
        </div>
      </div>
    </section>
  );
}
