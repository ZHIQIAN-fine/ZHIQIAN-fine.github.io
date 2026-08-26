# results 目录说明

- `eval_v0_baseline.json` / `eval_v1_cost_up.json` / `eval_v2_deck_mix.json`
  v2 那一轮（护盾提费 / 起手多样化）的 1000 局评估结果。

- `summary_all.md` / `summary_all.json`
  **v3 轮的最终汇总**：4 组（v0 基线 / v3a 掉落分层 / v3b 数值修正 / v3ab 叠加）
  × 3 个训练种子（42 / 7 / 123）× 1000 局确定性评估，外加 v1 / v2 历史锚点对照。
  原始逐组结果、图表与实验报告在 [`../../experiments-v3/`](../../experiments-v3/)。
