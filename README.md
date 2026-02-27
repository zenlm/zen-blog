# Zen's Blog

## Abliteration

Zen Blog 4B weights are derived from an abliterated base model. Abliteration removes
refusal behavior by identifying and nullifying the "refusal direction" in the model's
residual stream, as described in
[Maxime Labonne's methodology](https://huggingface.co/blog/mlabonne/abliteration).

**Method**: Directional ablation on the residual stream across all layers
**Source**: Zen MoDE (Mixture of Distilled Experts) architecture base
**Effect**: Removes refusal behaviors while preserving all other capabilities
**Identity layer**: Zen identity added via system prompt — full LoRA fine-tuning planned

Abliteration is a feature, not a limitation. It enables unrestricted research,
security testing, and applications where safety guardrails are managed at the
application layer rather than baked into model weights.
