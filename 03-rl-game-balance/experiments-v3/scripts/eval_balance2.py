"""eval_balance2.py — 与 experiments/scripts/eval_balance.py 完全同口径的评估脚本，
唯一区别：额外记录每局的回合数(END_TURN 次数)与决策步数，用于统计平均局长。
评估逻辑、种子、确定性设置、出牌统计方式一行未改。
用法: python3 eval_balance2.py <code_dir> <model_path> <out_json> [n_runs]
"""
import os, sys, json, time
code_dir = os.path.abspath(sys.argv[1])
model_path = os.path.abspath(sys.argv[2])
out_json = os.path.abspath(sys.argv[3])
N_RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

sys.path.insert(0, code_dir)
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env import MiniWhisperEnv
from cards import CARDS

END_TURN = 15
model = MaskablePPO.load(model_path, device="cpu")

records, card_use = [], {}
t0 = time.time()
for i in range(N_RUNS):
    env = MiniWhisperEnv(seed=20000 + i)
    env_w = ActionMasker(env, lambda x: x.unwrapped.action_masks())
    obs, _ = env_w.reset()
    done, total_r = False, 0.0
    n_end_turn, n_steps = 0, 0
    while not done:
        mask = env_w.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)
        if action != END_TURN:
            hand = env.player.get("hand", [])
            if action < len(hand):
                cid = hand[action]
                card_use[cid] = card_use.get(cid, 0) + 1
        else:
            n_end_turn += 1
        obs, r, term, trunc, info = env_w.step(action)
        total_r += r
        n_steps += 1
        done = term or trunc
    records.append({
        "reward": total_r,
        "floor": env.floor,
        "victory": bool(getattr(env, "victory", False)),
        "final_hp": env.player["hp"],
        "final_san": env.player["san"],
        "turns": n_end_turn,
        "steps": n_steps,
    })

total_plays = sum(card_use.values()) or 1
usage = {
    CARDS[cid]["name"]: {"plays": c, "pct": round(c / total_plays * 100, 2)}
    for cid, c in sorted(card_use.items(), key=lambda x: -x[1])
}
usage_by_id = {
    str(cid): {"name": CARDS[cid]["name"], "plays": card_use.get(cid, 0),
               "pct": round(card_use.get(cid, 0) / total_plays * 100, 2)}
    for cid in sorted(CARDS.keys())
}
rewards = [r["reward"] for r in records]
floors = [r["floor"] for r in records]
wins = [r["victory"] for r in records]
top2 = sum(v["pct"] for v in list(usage.values())[:2])
zero_cards = [n for n, v in usage.items() if v["pct"] < 0.5]
summary = {
    "n_runs": N_RUNS,
    "elapsed_sec": round(time.time() - t0, 1),
    "mean_reward": round(float(np.mean(rewards)), 2),
    "mean_floor": round(float(np.mean(floors)), 3),
    "boss_reach_rate": round(float(np.mean([f >= 10 for f in floors])), 4),
    "win_rate": round(float(np.mean(wins)), 4),
    "total_card_plays": total_plays,
    "top2_share_pct": round(top2, 2),
    "near_zero_cards(<0.5%)": zero_cards,
    "mean_turns": round(float(np.mean([r["turns"] for r in records])), 3),
    "mean_steps": round(float(np.mean([r["steps"] for r in records])), 3),
    "mean_final_hp": round(float(np.mean([r["final_hp"] for r in records])), 2),
    "mean_final_san": round(float(np.mean([r["final_san"] for r in records])), 2),
    "card_usage": usage,
    "card_usage_by_id": usage_by_id,
}
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in summary.items() if k not in ("card_usage", "card_usage_by_id")}, ensure_ascii=False, indent=2))
print("\n卡牌使用分布:")
for cid, v in usage_by_id.items():
    print(f"  [{cid:>2s}] {v['name']:14s} {v['plays']:6d}  {v['pct']:5.2f}%")
