import { useEffect, useMemo, useState } from "react";
import { getLoraTargetModules } from "../../api";
import { defaultLoraTargets } from "../../defaults";
import { useProject } from "../../state/ProjectContext";
import type { LoraTarget, Method } from "../../types";

export default function ModelStage() {
  const { project, updateModel } = useProject();
  const model = project.model;
  const [discoveredTargets, setDiscoveredTargets] = useState<string[]>(defaultLoraTargets);

  useEffect(() => {
    let cancelled = false;
    getLoraTargetModules()
      .then((targets) => {
        if (!cancelled && targets.length > 0) {
          setDiscoveredTargets(Array.from(new Set([...defaultLoraTargets, ...targets])));
        }
      })
      .catch(() => {
        // The backend may be offline during frontend-only development.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const loraTargets = useMemo(
    () => Array.from(new Set([...discoveredTargets, ...model.selectedTargets])),
    [discoveredTargets, model.selectedTargets]
  );

  function toggleTarget(target: LoraTarget, checked: boolean) {
    updateModel({
      selectedTargets: checked
        ? Array.from(new Set([...model.selectedTargets, target]))
        : model.selectedTargets.filter((item) => item !== target)
    });
  }

  function changeMethod(method: Method) {
    updateModel({ method });
  }

  return (
    <section className="stage-panel">
      <div className="stage-heading">
        <span className="stage-kicker">Stage 3</span>
        <h1>Model</h1>
        <p>Choose the model, optional Hugging Face token, and loading/training method.</p>
      </div>

      <div className="stage-form">
        <div className="field-grid two">
          <label className="field">
            <span>Model name or local path</span>
            <input
              value={model.modelName}
              onChange={(event) => updateModel({ modelName: event.target.value })}
              placeholder="meta-llama/... or C:\\models\\..."
            />
          </label>

          <label className="field">
            <span>Hugging Face token <em>optional</em></span>
            <input
              type="password"
              value={model.hfKey}
              onChange={(event) => updateModel({ hfKey: event.target.value })}
              placeholder="hf_..."
            />
          </label>
        </div>

        <div className="field">
          <span>Method</span>
          <div className="segmented" role="group" aria-label="Model method">
            {(["full", "lora", "adaptor"] as Method[]).map((method) => (
              <button
                key={method}
                type="button"
                className={model.method === method ? "active" : ""}
                onClick={() => changeMethod(method)}
              >
                {method}
              </button>
            ))}
          </div>
        </div>

        {model.method === "adaptor" && (
          <label className="field">
            <span>Adaptor path</span>
            <input
              value={model.adaptorPath}
              onChange={(event) => updateModel({ adaptorPath: event.target.value })}
              placeholder="/path/to/adaptor"
            />
          </label>
        )}

        {model.method === "lora" && (
          <div className="field">
            <div className="field-label-row">
              <span>LoRA target modules</span>
              <div className="selection-actions">
                <button
                  className="text-button"
                  type="button"
                  onClick={() => updateModel({ selectedTargets: [...loraTargets] })}
                >
                  Select all
                </button>
                <button
                  className="text-button"
                  type="button"
                  onClick={() => updateModel({ selectedTargets: [] })}
                >
                  Clear
                </button>
              </div>
            </div>

            <div className="check-row">
              {loraTargets.map((target) => (
                <label key={target}>
                  <input
                    type="checkbox"
                    checked={model.selectedTargets.includes(target)}
                    onChange={(event) => toggleTarget(target, event.target.checked)}
                  />
                  <span>{target}</span>
                </label>
              ))}
            </div>
            <small>
              The list is read from the backend OpenAPI schema when available, so newly added LoRA modules appear automatically.
            </small>
          </div>
        )}
      </div>
    </section>
  );
}
