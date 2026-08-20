import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
base = json.load(open(os.path.join(R, "eval_v0_baseline.json")))
cost = json.load(open(os.path.join(R, "eval_v1_cost_up.json")))
deck = json.load(open(os.path.join(R, "eval_v2_deck_mix.json")))
groups = [("原始版本", base, "#8892a6"), ("方案A 护盾提费", cost, "#d95f5f"), ("方案B 起手多样化", deck, "#3fa66a")]

cards = list(base["card_usage"].keys())
fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], wspace=0.28)

ax = fig.add_subplot(gs[0])
y = np.arange(len(cards)); h = 0.26
for i, (name, d, c) in enumerate(groups):
    vals = [d["card_usage"].get(k, {"pct": 0})["pct"] for k in cards]
    ax.barh(y + (1 - i) * h, vals, height=h, color=c, label=name, edgecolor="white", linewidth=0.5)
ax.set_yticks(y); ax.set_yticklabels(cards, fontsize=10.5)
ax.invert_yaxis()
ax.set_xlabel("出牌占比 (%)", fontsize=11)
ax.set_title("卡牌使用分布对比（各 1000 局确定性评估）", fontsize=13, pad=12)
ax.legend(fontsize=10.5, loc="lower right"); ax.grid(alpha=0.25, axis="x")

ax2 = fig.add_subplot(gs[1])
metrics = ["头部两张卡\n出牌占比", "通关率", "抵达 Boss 层"]
keys = ["top2_share_pct", "win_rate", "boss_reach_rate"]
x = np.arange(len(metrics)); w = 0.26
for i, (name, d, c) in enumerate(groups):
    vals = [d["top2_share_pct"], d["win_rate"] * 100, d["boss_reach_rate"] * 100]
    b = ax2.bar(x + (i - 1) * w, vals, width=w, color=c, label=name, edgecolor="white", linewidth=0.6)
    ax2.bar_label(b, fmt="%.1f", fontsize=9, padding=2)
ax2.set_xticks(x); ax2.set_xticklabels(metrics, fontsize=10.5)
ax2.set_ylabel("百分比 (%)", fontsize=11); ax2.set_ylim(0, 112)
ax2.set_title("关键指标对比", fontsize=13, pad=12)
ax2.grid(alpha=0.25, axis="y")
fig.suptitle("Mini Whisper 平衡性改动回归验证：起手牌组结构才是失衡根因", fontsize=15, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(HERE, "..", "..", "assets", "balance_validation.png"), dpi=160)
print("saved")
