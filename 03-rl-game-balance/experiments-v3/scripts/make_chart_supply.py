#!/usr/bin/env python3
"""make_chart_supply.py — 供给侧诊断图（抽牌份额 vs 出牌份额 / 抽到即打率）
数据来源: results/drawplay/drawplay_*.json（seed 42 模型，各 300 局）
输出: output/fig_v3_supply.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FP)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=FP).get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#c9cdd4"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DP = os.path.join(ROOT, "results", "drawplay")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

GROUPS = ["v0_repro", "v3a", "v3b", "v3ab"]
SHORT = {"v0_repro": "v0 基线", "v3a": "v3a 掉落分层",
         "v3b": "v3b 数值修正", "v3ab": "v3ab 合并"}
COLOR = {"v0_repro": "#8892a6", "v3a": "#4a80c4",
         "v3b": "#d98b3f", "v3ab": "#3fa66a"}

D = {g: json.load(open(os.path.join(DP, f"drawplay_{g}.json"))) for g in GROUPS}
CARDS = [(cid, r["name"]) for cid, r in D["v0_repro"]["by_card"].items()]
NAMES = [n for _, n in CARDS]

fig = plt.figure(figsize=(18, 9.6))
gs = fig.add_gridspec(2, 3, wspace=0.26, hspace=0.42, width_ratios=[1.25, 1.25, 1])

# --- (a) 每张卡的抽到次数/局 ---
axA = fig.add_subplot(gs[0, :2])
x = np.arange(len(NAMES))
w = 0.2
for i, g in enumerate(GROUPS):
    vals = [D[g]["by_card"][cid]["draws_per_game"] for cid, _ in CARDS]
    axA.bar(x + (i - 1.5) * w, vals, width=w, color=COLOR[g], label=SHORT[g],
            edgecolor="white", linewidth=0.5)
axA.set_xticks(x)
axA.set_xticklabels(NAMES, fontsize=9.5, rotation=28, ha="right")
axA.set_ylabel("平均抽到次数 / 局", fontsize=11)
axA.set_yscale("log")
axA.set_title("(a) 供给侧：起手的两张牌每局被抽到约 50 次，其余 13 张全部在 1–6 次量级"
              "（对数轴）", fontsize=12.5, pad=10, loc="left")
axA.legend(fontsize=9.5, ncol=4)
axA.grid(alpha=0.22, axis="y")

# --- (b) top2 抽牌份额 vs 出牌份额 ---
axB = fig.add_subplot(gs[0, 2])
draw_share, play_share = [], []
for g in GROUPS:
    bc = D[g]["by_card"]
    td = sum(r["draws"] for r in bc.values())
    tp = sum(r["plays"] for r in bc.values())
    top2 = ["0", "3"]  # 生锈匕首 / 离子护盾
    draw_share.append(sum(bc[c]["draws"] for c in top2) / td * 100)
    play_share.append(sum(bc[c]["plays"] for c in top2) / tp * 100)
xi = np.arange(len(GROUPS))
b1 = axB.bar(xi - 0.19, draw_share, width=0.38, color="#a8b3c4", label="占全部抽牌")
b2 = axB.bar(xi + 0.19, play_share, width=0.38, color="#d95f5f", label="占全部出牌")
axB.bar_label(b1, fmt="%.1f", fontsize=9)
axB.bar_label(b2, fmt="%.1f", fontsize=9)
axB.set_xticks(xi)
axB.set_xticklabels([SHORT[g] for g in GROUPS], fontsize=9.5, rotation=12)
axB.set_ylabel("占比 (%)", fontsize=11)
axB.set_ylim(0, 100)
axB.set_title("(b) 出牌集中度 ≈ 抽牌集中度\n集中是供给问题，不是强度问题",
              fontsize=12.5, pad=10, loc="left")
axB.legend(fontsize=9.5)
axB.grid(alpha=0.22, axis="y")

# --- (c) 抽到即打率 ---
axC = fig.add_subplot(gs[1, :2])
for i, g in enumerate(GROUPS):
    vals = [D[g]["by_card"][cid]["play_rate_pct"] or 0 for cid, _ in CARDS]
    axC.bar(x + (i - 1.5) * w, vals, width=w, color=COLOR[g], label=SHORT[g],
            edgecolor="white", linewidth=0.5)
axC.axhline(60, color="#8c6d1f", ls="--", lw=1.2)
axC.text(0.2, 62, "能量预算决定的上限：3 能量 / 回合、抽 5 张 → 约 3/5 = 60%",
         fontsize=9, color="#8c6d1f")
axC.set_xticks(x)
axC.set_xticklabels(NAMES, fontsize=9.5, rotation=28, ha="right")
axC.set_ylabel("抽到即打率 (%)", fontsize=11)
axC.set_ylim(0, 78)
axC.set_title("(c) 机制刚需：在「每局抽到 40 次以上」的两张高供给卡里，"
              "护盾的转化率始终比匕首高 10 个百分点以上（最右三张卡样本极小，不作解读）",
              fontsize=12, pad=10, loc="left")
axC.legend(fontsize=9.5, ncol=4)
axC.grid(alpha=0.22, axis="y")

# --- (d) 两张核心牌的转化率对比 ---
axD = fig.add_subplot(gs[1, 2])
xi = np.arange(len(GROUPS))
shield = [D[g]["by_card"]["3"]["play_rate_pct"] for g in GROUPS]
dagger = [D[g]["by_card"]["0"]["play_rate_pct"] for g in GROUPS]
b1 = axD.bar(xi - 0.19, shield, width=0.38, color="#4a80c4", label="离子护盾（护甲每回合清零）")
b2 = axD.bar(xi + 0.19, dagger, width=0.38, color="#8892a6", label="生锈匕首（伤害会留存）")
axD.bar_label(b1, fmt="%.1f", fontsize=9)
axD.bar_label(b2, fmt="%.1f", fontsize=9)
axD.set_xticks(xi)
axD.set_xticklabels([SHORT[g] for g in GROUPS], fontsize=9.5, rotation=12)
axD.set_ylabel("抽到即打率 (%)", fontsize=11)
axD.set_ylim(0, 62)
axD.set_title("(d) 同样的供给量，护盾的转化率\n系统性高于匕首", fontsize=12.5, pad=10, loc="left")
axD.legend(fontsize=8.6)
axD.grid(alpha=0.22, axis="y")

fig.suptitle("供给侧诊断：为什么调数值和调掉落都动不了集中度　|　"
             "seed 42 模型 × 各 300 局，同时记录每张卡的抽到与打出次数",
             fontsize=15.5, y=0.975)
plt.savefig(os.path.join(OUT, "fig_v3_supply.png"), dpi=155, bbox_inches="tight",
            facecolor="white")
print("saved", os.path.join(OUT, "fig_v3_supply.png"))
