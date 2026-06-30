# Resources

## Unlearning - Hirundo.io

- Website: https://www.hirundo.io/
- Blog: https://www.hirundo.io/blog
- Relevant example: https://www.hirundo.io/blog/aya-expanse-32b-unlearning
- Relevant example: https://www.hirundo.io/blog/llama4-debiased

Hirundo positions its product as a machine unlearning platform for fixing risky model behavior at the model level instead of relying only on external guardrails or full retraining.

Tool functionality notes from the site and blog:

- Detect risky knowledge and behavior with built-in evaluations and red-team testing.
- Target unwanted behavior through machine unlearning by modifying the model parameters or internal representations associated with that behavior.
- Produce an updated model faster than full retraining while trying to preserve model utility.
- Support behavior unlearning for prompt injection vulnerabilities, bias, hallucinations, toxicity, and harmful outputs.
- Support data-focused unlearning use cases such as memorized PII and other data leakage concerns.
- Evaluate results with benchmarks such as Purple Llama for prompt injection/security and BBQ for bias.
- Present before-and-after model behavior so users can inspect what changed.
- Provide a platform workflow where users can select undesirable behaviors to unlearn.

Ideas to consider for this repo:

- A dedicated "risk target" selector for bias, toxicity, hallucination, PII, prompt injection, or custom forget data.
- Side-by-side before/after evaluation results.
- Clear separation between dataset upload, unlearning configuration, evaluation, and final model export.
- Benchmark-driven result summaries instead of only raw training output.

## Unsloth Studio

- Website: https://unsloth.ai/
- Studio: https://unsloth.ai/studio

Use Unsloth Studio as UI inspiration for a focused model-training workflow. The useful reference point is the product shape: a practical interface for configuring model work without making users reason through every low-level training detail.

Ideas to consider for this repo:

- Keep the first screen as the actual training console, not a marketing page.
- Use compact controls for model, dataset, GPU, and hyperparameter setup.
- Show validation feedback near the controls that caused it.
- Keep the generated config visible and copyable.
- Make long-running training states obvious with progress, logs, and final artifacts.
