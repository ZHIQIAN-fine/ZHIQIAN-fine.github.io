#!/usr/bin/env python3
"""make_chart_v3.py — v3 分组对照实验可视化
输出:
  output/fig_v3_main.png        主图: 逐卡分布 / 关键指标(带种子误差棒) / 通关率-集中度散点 / 训练曲线
  output/fig_v3_diagnosis.png   诊断图: 集中度构成 / 被修卡命运 / 难度与局长
用法: python3 make_chart_v3.py
"""
import json, os, re
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
RES = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

S = json.load(open(os.path.join(RES, "summary_all.json")))
GROUPS = ["v0_repro", "v3a", "v3b", "v3ab"]
SHORT = {"v0_repro": "v0 基线", "v3a": "v3a 掉落分层",
         "v3b": "v3b 数值修正", "v3ab": "v3ab 合并"}
COLOR = {"v0_repro": "#8892a6", "v3a": "#4a80c4",
         "v3b": "#d98b3f", "v3ab": "#3fa66a"}
SEEDS = ["42", "7", "123"]
CARD_ORDER = ["生锈匕首", "离子护盾", "触发链×2", "进程冻结", "冥想程序", "代码污染",
              "OVERCLOCK", "404·死灵协议", "过载电浆刃", "不该看的日志", "拆解",
              "格式化记忆", "亵渎协议", "无名之吻", "黑色低语"]


def agg(g, k, field="mean"):
    a = S["groups"][g]["agg"].get(k)
    return None if a is None else a[field]


def seed_vals(g, k):
    return [S["groups"][g]["seeds"][s][k] for s in SEEDS if s in S["groups"][g]["seeds"]]


def parse_curve(path, key):
    """从训练日志里按顺序抽取 rollout 指标序列 + total_timesteps。"""
    if not os.path.exists(path):
        return [], []
    txt = open(path, errors="ignore").read()
    vals = [float(v) for v in re.findall(rf"\|\s+{key}\s+\|\s+([-\d.e+]+)\s+\|", txt)]
    steps = [int(v) for v in re.findall(r"\|\s+total_timesteps\s+\|\s+(\d+)\s+\|", txt)]
    n = min(len(vals), len(steps))
    return steps[:n], vals[:n]


def log_path(g, s):
    base = "train_v0_repro" if g == "v0_repro" else f"train_{g}"
    return os.path.join(LOGS, f"{base}.log" if s == "42" else f"{base}_s{s}.log")


# ============================ 主图 ============================
fig = plt.figure(figsize=(17.5, 11))
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1.25, 1],
                      wspace=0.24, hspace=0.32)

# --- (a) 逐卡出牌占比 ---
ax = fig.add_subplot(gs[:, 0])
y = np.arange(len(CARD_ORDER))
h = 0.19
for i, g in enumerate(GROUPS):
    pct = S["groups"][g]["agg"]["pct_by_name"]
    vals = [pct.get(c, 0) for c in CARD_ORDER]
    ax.barh(y + (1.5 - i) * h, vals, height=h, color=COLOR[g],
            label=SHORT[g], edgecolor="white", linewidth=0.5)
v2 = S["reference"].get("v2_deck_mix", {}).get("pct_by_name", {})
if v2:
    ax.scatter([v2.get(c, 0) for c in CARD_ORDER], y, s=34, marker="D",
               facecolor="none", edgecolor="#b0447a", linewidth=1.5, zorder=5,
               label="v2 起手多样化（历史唯一有效解）")
ax.axvline(0.5, color="#d95f5f", ls=":", lw=1.2)
ax.text(0.62, len(CARD_ORDER) - 0.4, "0.5% 死卡线", color="#d95f5f", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(CARD_ORDER, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("出牌占比 (%)　·　3 种子均值", fontsize=11)
ax.set_title("(a) 逐卡出牌分布：两张核心牌的地位没有被任何一组改动动摇",
             fontsize=13, pad=10, loc="left")
ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
ax.grid(alpha=0.25, axis="x")

# --- (b) 关键指标 + 种子极差误差棒 ---
ax2 = fig.add_subplot(gs[0, 1])
metrics = [("win_rate", "通关率", 100), ("boss_reach_rate", "Boss 抵达率", 100),
           ("top2_share_pct", "头部两卡占比", 1)]
x = np.arange(len(metrics))
w = 0.2
for i, g in enumerate(GROUPS):
    means, lo, hi = [], [], []
    for k, _, sc in metrics:
        vs = [v * sc for v in seed_vals(g, k)]
        m = float(np.mean(vs))
        means.append(m)
        lo.append(m - min(vs))
        hi.append(max(vs) - m)
    b = ax2.bar(x + (i - 1.5) * w, means, width=w, color=COLOR[g],
                label=SHORT[g], edgecolor="white", linewidth=0.6,
                yerr=[lo, hi], capsize=3, error_kw=dict(lw=1, ecolor="#5a5a5a"))
    ax2.bar_label(b, fmt="%.1f", fontsize=8.5, padding=6)
ax2.set_xticks(x)
ax2.set_xticklabels([m[1] for m in metrics], fontsize=11)
ax2.set_ylabel("百分比 (%)", fontsize=11)
ax2.set_ylim(0, 118)
ax2.set_title("(b) 关键指标（误差棒 = 3 种子极差）", fontsize=13, pad=10, loc="left")
ax2.legend(fontsize=9, ncol=2, loc="upper center")
ax2.grid(alpha=0.25, axis="y")

# --- (c) 通关率 vs 集中度 散点 ---
ax3 = fig.add_subplot(gs[1, 1])
for g in GROUPS:
    xs = [S["groups"][g]["seeds"][s]["top2_share_pct"] for s in SEEDS if s in S["groups"][g]["seeds"]]
    ys = [S["groups"][g]["seeds"][s]["win_rate"] * 100 for s in SEEDS if s in S["groups"][g]["seeds"]]
    ax3.scatter(xs, ys, s=110, color=COLOR[g], edgecolor="white", linewidth=1.2,
                label=SHORT[g], zorder=4)
    ax3.scatter([np.mean(xs)], [np.mean(ys)], s=260, marker="*",
                color=COLOR[g], edgecolor="#333", linewidth=0.8, zorder=5)
ref = [("v2_deck_mix", "v2 起手多样化", "#b0447a", "D"),
       ("v1_cost_up", "v1 护盾提费", "#d95f5f", "X")]
for key, lbl, c, mk in ref:
    r = S["reference"].get(key)
    if r:
        ax3.scatter([r["top2_share_pct"]], [r["win_rate"] * 100], s=150, marker=mk,
                    color=c, edgecolor="white", linewidth=1.2, label=lbl, zorder=6)
ax3.set_xlabel("头部两卡出牌占比 (%)　←　更健康", fontsize=11)
ax3.set_ylabel("通关率 (%)", fontsize=11)
ax3.set_title("(c) 只有 v2 把集中度打下来；v3 三组全在右侧原地不动",
              fontsize=12.5, pad=10, loc="left")
ax3.legend(fontsize=9, loc="lower left", ncol=2)
ax3.grid(alpha=0.25)
ax3.axvspan(70, 85, color="#d95f5f", alpha=0.06)

fig.suptitle("v3 分组对照：掉落池分层(v3a) vs 劣势卡数值修正(v3b) vs 合并(v3ab)　"
             "|　4 组 × 3 种子 × 1000 局确定性评估",
             fontsize=15.5, y=0.975)
plt.savefig(os.path.join(OUT, "fig_v3_main.png"), dpi=155, bbox_inches="tight",
            facecolor="white")
print("saved", os.path.join(OUT, "fig_v3_main.png"))
plt.close(fig)

# ============================ 诊断图 ============================
fig2 = plt.figure(figsize=(17.5, 9.5))
gs2 = fig2.add_gridspec(2, 3, wspace=0.28, hspace=0.38)

# --- (a) 集中度构成堆叠 ---
axA = fig2.add_subplot(gs2[0, 0])
labels, shield, dagger, rest = [], [], [], []
order = [("v0_repro", None), ("v3a", None), ("v3b", None), ("v3ab", None)]
for g, _ in order:
    labels.append(SHORT[g])
    p = S["groups"][g]["agg"]["pct_by_name"]
    shield.append(p["离子护盾"])
    dagger.append(p["生锈匕首"])
    rest.append(100 - p["离子护盾"] - p["生锈匕首"])
v2p = S["reference"].get("v2_deck_mix", {}).get("pct_by_name")
if v2p:
    labels.append("v2 起手\n多样化")
    shield.append(v2p["离子护盾"])
    dagger.append(v2p["生锈匕首"])
    rest.append(100 - v2p["离子护盾"] - v2p["生锈匕首"])
xi = np.arange(len(labels))
axA.bar(xi, shield, color="#4a80c4", label="离子护盾（1 费 5 甲，护甲回合清零）")
axA.bar(xi, dagger, bottom=shield, color="#8892a6", label="生锈匕首（1 费 6 伤）")
axA.bar(xi, rest, bottom=np.array(shield) + np.array(dagger), color="#e3e6eb",
        label="其余 13 张合计")
for i in range(len(labels)):
    axA.text(i, shield[i] / 2, f"{shield[i]:.0f}", ha="center", va="center",
             fontsize=9.5, color="white")
    axA.text(i, shield[i] + dagger[i] / 2, f"{dagger[i]:.0f}", ha="center",
             va="center", fontsize=9.5, color="white")
    axA.text(i, shield[i] + dagger[i] + rest[i] / 2, f"{rest[i]:.0f}", ha="center",
             va="center", fontsize=9.5, color="#4e5969")
axA.set_xticks(xi)
axA.set_xticklabels(labels, fontsize=9, rotation=16, ha="right")
axA.set_ylabel("出牌占比 (%)", fontsize=10.5)
axA.set_ylim(0, 105)
axA.set_title("诊断一 · 机制刚需\n护甲每回合清零 → 护盾必须重复打", fontsize=12, pad=8, loc="left")
axA.legend(fontsize=8.2, loc="upper right", framealpha=0.95)
axA.grid(alpha=0.2, axis="y")

# --- (b) 获取曲线：掉落层卡的占比 ---
axB = fig2.add_subplot(gs2[0, 1])
late = ["不该看的日志", "拆解", "格式化记忆", "亵渎协议"]
xi = np.arange(len(late))
w = 0.2
for i, g in enumerate(GROUPS):
    p = S["groups"][g]["agg"]["pct_by_name"]
    vals = [p.get(c, 0) for c in late]
    b = axB.bar(xi + (i - 1.5) * w, vals, width=w, color=COLOR[g], label=SHORT[g],
                edgecolor="white", linewidth=0.5)
    axB.bar_label(b, fmt="%.1f", fontsize=7.5, padding=2)
axB.axhline(0.5, color="#d95f5f", ls=":", lw=1.2)
axB.set_xticks(xi)
axB.set_xticklabels(late, fontsize=9.5)
axB.set_ylabel("出牌占比 (%)", fontsize=10.5)
axB.set_ylim(0, 3.9)
axB.set_title("诊断二 · 获取曲线\n晚档掉落卡（v3a 推到 7–10 层才放）", fontsize=12, pad=8, loc="left")
axB.legend(fontsize=8.2, ncol=2, loc="upper right")
axB.grid(alpha=0.2, axis="y")

# --- (c) 费用失衡：被 v3b 改数值的四张卡 ---
axC = fig2.add_subplot(gs2[0, 2])
fixed = ["404·死灵协议", "代码污染", "OVERCLOCK", "亵渎协议"]
xi = np.arange(len(fixed))
for i, g in enumerate(GROUPS):
    p = S["groups"][g]["agg"]["pct_by_name"]
    vals = [p.get(c, 0) for c in fixed]
    b = axC.bar(xi + (i - 1.5) * w, vals, width=w, color=COLOR[g], label=SHORT[g],
                edgecolor="white", linewidth=0.5)
    axC.bar_label(b, fmt="%.1f", fontsize=7.5, padding=2)
axC.axhline(0.5, color="#d95f5f", ls=":", lw=1.2)
axC.set_xticks(xi)
axC.set_xticklabels(fixed, fontsize=9.5)
axC.set_ylabel("出牌占比 (%)", fontsize=10.5)
axC.set_ylim(0, 5.2)
axC.set_title("诊断三 · 费用失衡\nv3b 改数值的四张严格劣势卡", fontsize=12, pad=8, loc="left")
axC.legend(fontsize=8.2, ncol=2, loc="upper right")
axC.grid(alpha=0.2, axis="y")

# --- (d) 通关率逐种子 ---
axD = fig2.add_subplot(gs2[1, 0])
for i, g in enumerate(GROUPS):
    vs = [v * 100 for v in seed_vals(g, "win_rate")]
    axD.scatter([i] * len(vs), vs, s=90, color=COLOR[g], edgecolor="white",
                linewidth=1, zorder=4)
    axD.plot([i - 0.28, i + 0.28], [np.mean(vs)] * 2, color=COLOR[g], lw=2.5, zorder=3)
    axD.text(i, max(vs) + 2.2, f"{np.mean(vs):.1f}%", ha="center", fontsize=10,
             color=COLOR[g], fontweight="bold")
axD.set_xticks(range(len(GROUPS)))
axD.set_xticklabels([SHORT[g] for g in GROUPS], fontsize=10)
axD.set_ylabel("通关率 (%)", fontsize=10.5)
axD.set_title("(d) 通关率：逐种子点 + 组均值横线", fontsize=12, pad=8, loc="left")
axD.grid(alpha=0.2, axis="y")

# --- (e) 局长与回合数 ---
axE = fig2.add_subplot(gs2[1, 1])
xi = np.arange(len(GROUPS))
steps = [agg(g, "mean_steps") for g in GROUPS]
turns = [agg(g, "mean_turns") for g in GROUPS]
b1 = axE.bar(xi - 0.19, steps, width=0.38, color="#4a80c4", label="平均决策步数")
b2 = axE.bar(xi + 0.19, turns, width=0.38, color="#d98b3f", label="平均回合数")
axE.bar_label(b1, fmt="%.1f", fontsize=8.5)
axE.bar_label(b2, fmt="%.1f", fontsize=8.5)
axE.set_xticks(xi)
axE.set_xticklabels([SHORT[g] for g in GROUPS], fontsize=10)
axE.set_title("(e) 局长：v3b 更快解决战斗", fontsize=12, pad=8, loc="left")
axE.legend(fontsize=9)
axE.grid(alpha=0.2, axis="y")

# --- (f) 训练曲线 ep_rew_mean ---
axF = fig2.add_subplot(gs2[1, 2])
for g in GROUPS:
    for j, s in enumerate(SEEDS):
        st, vs = parse_curve(log_path(g, s), "ep_rew_mean")
        if not st:
            continue
        axF.plot(st, vs, color=COLOR[g], alpha=0.75 if j == 0 else 0.4,
                 lw=1.6 if j == 0 else 1.0,
                 label=SHORT[g] if j == 0 else None)
axF.set_xlabel("训练步数", fontsize=10.5)
axF.set_ylabel("ep_rew_mean", fontsize=10.5)
axF.set_title("(f) 训练回报曲线（每组 3 种子）", fontsize=12, pad=8, loc="left")
axF.legend(fontsize=9)
axF.grid(alpha=0.2)

fig2.suptitle("三层诊断：机制刚需 / 获取曲线 / 费用失衡　—　"
              "改数值能救活劣势卡与难度曲线，但救不了结构性集中度",
              fontsize=15.5, y=0.975)
plt.savefig(os.path.join(OUT, "fig_v3_diagnosis.png"), dpi=155, bbox_inches="tight",
            facecolor="white")
print("saved", os.path.join(OUT, "fig_v3_diagnosis.png"))
