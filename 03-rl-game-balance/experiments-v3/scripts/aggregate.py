#!/usr/bin/env python3
"""aggregate.py — 汇总 v0/v3a/v3b/v3ab 四组 × 三种子(42/7/123)的 1000 局评估结果，
外加初版原权重基线与 v1/v2 历史结果，产出：
  - results/summary_all.json   机器可读汇总(逐组逐种子 + 组内均值/极差)
  - results/summary_all.md     人读对比表(通关率/Boss抵达/集中度/逐卡频率/局长/回报)
用法: python3 aggregate.py
"""
import json, os, re, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
HIST = os.path.abspath(os.path.join(
    ROOT, "..", "ZHIQIAN-fine.github.io", "03-rl-game-balance", "experiments", "results"))

GROUPS = ["v0_repro", "v3a", "v3b", "v3ab"]
GROUP_LABEL = {
    "v0_repro": "v0 基线复现",
    "v3a": "v3a 掉落池分层",
    "v3b": "v3b 劣势卡数值修正",
    "v3ab": "v3ab 两者合并",
}
SEEDS = ["42", "7", "123"]

CARD_ORDER = [
    "生锈匕首", "过载电浆刃", "触发链×2", "离子护盾", "冥想程序", "404·死灵协议",
    "进程冻结", "代码污染", "OVERCLOCK", "不该看的日志", "拆解", "格式化记忆",
    "亵渎协议", "无名之吻", "黑色低语",
]

METRICS = [
    ("win_rate", "通关率", 100, "%"),
    ("boss_reach_rate", "Boss 抵达率", 100, "%"),
    ("top2_share_pct", "头部两卡出牌占比", 1, "%"),
    ("shield_pct", "离子护盾单卡占比", 1, "%"),
    ("dagger_pct", "生锈匕首单卡占比", 1, "%"),
    ("mean_reward", "评估平均奖励", 1, ""),
    ("mean_floor", "平均到达层数", 1, ""),
    ("mean_steps", "平均决策步数(局长)", 1, ""),
    ("mean_turns", "平均回合数", 1, ""),
    ("mean_final_hp", "终局 HP", 1, ""),
    ("mean_final_san", "终局 SAN", 1, ""),
    ("n_dead_cards", "出牌占比<0.5% 的卡(张)", 1, ""),
]


def eval_path(group, seed):
    return os.path.join(RES, f"eval_{group}.json" if seed == "42"
                        else f"eval_{group}_s{seed}.json")


def load_eval(path):
    with open(path) as f:
        d = json.load(f)
    usage = d.get("card_usage", {})
    d["shield_pct"] = usage.get("离子护盾", {}).get("pct", 0.0)
    d["dagger_pct"] = usage.get("生锈匕首", {}).get("pct", 0.0)
    d["n_dead_cards"] = len(d.get("near_zero_cards(<0.5%)", []))
    d["pct_by_name"] = {n: v["pct"] for n, v in usage.items()}
    return d


def parse_train_log(path):
    """取训练日志里最后一次 rollout 的 ep_len_mean / ep_rew_mean。"""
    if not os.path.exists(path):
        return {}
    txt = open(path, errors="ignore").read()
    out = {}
    for key, field in (("ep_len_mean", "ep_len_mean"), ("ep_rew_mean", "ep_rew_mean")):
        vals = re.findall(rf"\|\s+{key}\s+\|\s+([-\d.e+]+)\s+\|", txt)
        if vals:
            out[field] = float(vals[-1])
    fps = re.findall(r"\|\s+fps\s+\|\s+([\d.]+)\s+\|", txt)
    if fps:
        out["fps_last"] = float(fps[-1])
    return out


def train_log_path(group, seed):
    base = "train_v0_repro" if group == "v0_repro" else f"train_{group}"
    return os.path.join(LOGS, f"{base}.log" if seed == "42" else f"{base}_s{seed}.log")


def fmt(v, scale=1, unit="", nd=2):
    if v is None:
        return "—"
    return f"{v * scale:.{nd}f}{unit}"


def main():
    data = {"groups": {}, "reference": {}}

    # 四组 × 三种子
    for g in GROUPS:
        data["groups"][g] = {"label": GROUP_LABEL[g], "seeds": {}, "agg": {}}
        for s in SEEDS:
            p = eval_path(g, s)
            if not os.path.exists(p):
                print(f"[warn] 缺少 {p}")
                continue
            d = load_eval(p)
            tl = parse_train_log(train_log_path(g, s))
            rec = {k: d.get(k) for k, *_ in METRICS}
            rec["pct_by_name"] = d["pct_by_name"]
            rec["near_zero"] = d.get("near_zero_cards(<0.5%)", [])
            rec["total_card_plays"] = d.get("total_card_plays")
            rec.update(tl)
            data["groups"][g]["seeds"][s] = rec

    # 组内聚合(均值/最小/最大)
    agg_keys = [k for k, *_ in METRICS] + ["ep_len_mean", "ep_rew_mean"]
    for g in GROUPS:
        seeds = data["groups"][g]["seeds"]
        if not seeds:
            continue
        for k in agg_keys:
            vals = [r[k] for r in seeds.values() if r.get(k) is not None]
            if not vals:
                continue
            data["groups"][g]["agg"][k] = {
                "mean": round(st.mean(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "sd": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        # 逐卡占比组内均值
        card_mean = {}
        for name in CARD_ORDER:
            vals = [r["pct_by_name"].get(name, 0.0) for r in seeds.values()]
            card_mean[name] = round(st.mean(vals), 2)
        data["groups"][g]["agg"]["pct_by_name"] = card_mean

    # 参考基线：初版原权重 + v1/v2 历史
    ow = os.path.join(RES, "eval_v0_originalweights.json")
    if os.path.exists(ow):
        d = load_eval(ow)
        data["reference"]["v0_original_weights"] = {
            "label": "v0 初版原权重(口径校准锚点)",
            **{k: d.get(k) for k, *_ in METRICS},
            "pct_by_name": d["pct_by_name"],
        }
    for fn, label in (("eval_v1_cost_up.json", "v1 护盾提费(历史)"),
                      ("eval_v2_deck_mix.json", "v2 起手多样化(历史)")):
        p = os.path.join(HIST, fn)
        if os.path.exists(p):
            d = load_eval(p)
            data["reference"][fn.replace("eval_", "").replace(".json", "")] = {
                "label": label,
                **{k: d.get(k) for k, *_ in METRICS},
                "pct_by_name": d["pct_by_name"],
            }

    with open(os.path.join(RES, "summary_all.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # ---------------- Markdown ----------------
    L = []
    L.append("# v3 分组对照实验汇总（掉落池分层 vs 劣势卡数值修正）\n")
    L.append("每组 3 个训练种子（42 / 7 / 123），每个模型 1000 局确定性评估（种子 20000–20999，训练未见过）。\n")

    L.append("\n## 1. 主表：组内均值（3 种子）\n")
    head = "| 指标 | " + " | ".join(GROUP_LABEL[g] for g in GROUPS) + " |"
    L.append(head)
    L.append("|---|" + "---|" * len(GROUPS))
    for k, label, scale, unit in METRICS:
        row = [label]
        for g in GROUPS:
            a = data["groups"][g]["agg"].get(k)
            row.append("—" if not a else f"{a['mean'] * scale:.2f}{unit}")
        L.append("| " + " | ".join(row) + " |")
    for k, label in (("ep_len_mean", "训练末 ep_len_mean"), ("ep_rew_mean", "训练末 ep_rew_mean")):
        row = [label]
        for g in GROUPS:
            a = data["groups"][g]["agg"].get(k)
            row.append("—" if not a else f"{a['mean']:.2f}")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## 2. 种子稳健性：逐种子明细\n")
    L.append("| 组别 | 种子 | 通关率 | Boss 抵达 | 头部两卡占比 | 护盾占比 | 平均奖励 | 局长(步) | ep_len_mean | ep_rew_mean |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for g in GROUPS:
        for s in SEEDS:
            r = data["groups"][g]["seeds"].get(s)
            if not r:
                continue
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                GROUP_LABEL[g], s,
                fmt(r["win_rate"], 100, "%"), fmt(r["boss_reach_rate"], 100, "%"),
                fmt(r["top2_share_pct"], 1, "%"), fmt(r["shield_pct"], 1, "%"),
                fmt(r["mean_reward"]), fmt(r["mean_steps"]),
                fmt(r.get("ep_len_mean")), fmt(r.get("ep_rew_mean"))))

    L.append("\n### 组内极差（max − min，用于判断差异是否超出种子噪声）\n")
    L.append("| 指标 | " + " | ".join(GROUP_LABEL[g] for g in GROUPS) + " |")
    L.append("|---|" + "---|" * len(GROUPS))
    for k, label, scale, unit in METRICS[:6]:
        row = [label]
        for g in GROUPS:
            a = data["groups"][g]["agg"].get(k)
            row.append("—" if not a else f"{(a['max'] - a['min']) * scale:.2f}{unit}")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## 3. 逐卡出牌占比（组内 3 种子均值，%）\n")
    L.append("| 卡牌 | 费 | " + " | ".join(GROUP_LABEL[g] for g in GROUPS) +
             " | v2 起手多样化(历史) |")
    L.append("|---|---|" + "---|" * (len(GROUPS) + 1))
    costs = {"生锈匕首": 1, "过载电浆刃": 2, "触发链×2": 1, "离子护盾": 1, "冥想程序": 1,
             "404·死灵协议": 2, "进程冻结": 1, "代码污染": 1, "OVERCLOCK": 2,
             "不该看的日志": 0, "拆解": 0, "格式化记忆": 1, "亵渎协议": 2,
             "无名之吻": 1, "黑色低语": 0}
    v2 = data["reference"].get("v2_deck_mix", {}).get("pct_by_name", {})
    for name in CARD_ORDER:
        row = [name, str(costs.get(name, ""))]
        for g in GROUPS:
            row.append(f"{data['groups'][g]['agg']['pct_by_name'].get(name, 0):.2f}")
        row.append(f"{v2.get(name, 0):.2f}" if v2 else "—")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## 4. 历史与锚点对照\n")
    L.append("| 组别 | 通关率 | Boss 抵达 | 头部两卡占比 | 护盾占比 | 平均奖励 | 死卡数 |")
    L.append("|---|---|---|---|---|---|---|")
    for key, r in data["reference"].items():
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["label"], fmt(r["win_rate"], 100, "%"), fmt(r["boss_reach_rate"], 100, "%"),
            fmt(r["top2_share_pct"], 1, "%"), fmt(r["shield_pct"], 1, "%"),
            fmt(r["mean_reward"]), r["n_dead_cards"]))
    for g in GROUPS:
        a = data["groups"][g]["agg"]
        if not a:
            continue
        L.append("| {}（3 种子均值） | {} | {} | {} | {} | {} | {} |".format(
            GROUP_LABEL[g],
            fmt(a["win_rate"]["mean"], 100, "%"), fmt(a["boss_reach_rate"]["mean"], 100, "%"),
            fmt(a["top2_share_pct"]["mean"], 1, "%"), fmt(a["shield_pct"]["mean"], 1, "%"),
            fmt(a["mean_reward"]["mean"]), f"{a['n_dead_cards']['mean']:.1f}"))

    with open(os.path.join(RES, "summary_all.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\n[written] {RES}/summary_all.json")
    print(f"[written] {RES}/summary_all.md")


if __name__ == "__main__":
    main()
