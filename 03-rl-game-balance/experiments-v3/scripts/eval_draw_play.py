#!/usr/bin/env python3
"""eval_draw_play.py — 逐卡「抽到即打率」测量
出牌占比只能说明「打了多少」，无法区分两种成因：
  (1) 机制刚需：这张卡抽到就必须打（护甲每回合清零 → 护盾不打就浪费）
  (2) 供给多：牌组里这张卡本来就多
本脚本同时统计每张卡的「被抽到次数」与「被打出次数」，
抽到即打率 = 打出 / 抽到。接近 100% 即为刚需牌（智能体没有取舍空间）。

用法: python3 eval_draw_play.py <code_dir> <model.zip> <out_json> [n_runs]
"""
import os, sys, json, collections
code_dir = os.path.abspath(sys.argv[1])
model_path = os.path.abspath(sys.argv[2])
out_json = os.path.abspath(sys.argv[3])
N_RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 300

sys.path.insert(0, code_dir)
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env import MiniWhisperEnv
from cards import CARDS

END_TURN = 15
model = MaskablePPO.load(model_path, device="cpu")

draws = collections.Counter()
plays = collections.Counter()

for i in range(N_RUNS):
    env = MiniWhisperEnv(seed=20000 + i)

    # 记录抽牌：包裹 _draw，用手牌多重集差分统计新增卡
    orig_draw = env._draw

    def patched_draw(n, _env=env, _orig=orig_draw):
        before = collections.Counter(_env.player["hand"])
        _orig(n)
        after = collections.Counter(_env.player["hand"])
        for cid, c in (after - before).items():
            draws[cid] += c
    env._draw = patched_draw

    env_w = ActionMasker(env, lambda x: x.unwrapped.action_masks())
    obs, _ = env_w.reset()
    done = False
    while not done:
        mask = env_w.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)
        if action != END_TURN:
            hand = env.player.get("hand", [])
            if action < len(hand):
                plays[hand[action]] += 1
        obs, r, term, trunc, info = env_w.step(action)
        done = term or trunc

rows = {}
for cid in sorted(CARDS.keys()):
    d, p = draws.get(cid, 0), plays.get(cid, 0)
    rows[str(cid)] = {
        "name": CARDS[cid]["name"],
        "cost": CARDS[cid]["cost"],
        "draws": d,
        "plays": p,
        "draws_per_game": round(d / N_RUNS, 2),
        "plays_per_game": round(p / N_RUNS, 2),
        "play_rate_pct": round(p / d * 100, 2) if d else None,
    }
out = {"n_runs": N_RUNS, "model": os.path.basename(model_path),
       "code_dir": os.path.basename(code_dir), "by_card": rows}
with open(out_json, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"{'卡牌':16s} {'费':>2s} {'抽/局':>7s} {'打/局':>7s} {'抽到即打率':>9s}")
for cid, r in rows.items():
    pr = "—" if r["play_rate_pct"] is None else f"{r['play_rate_pct']:6.1f}%"
    print(f"[{cid:>2s}] {r['name']:14s} {r['cost']:>2d} {r['draws_per_game']:7.2f} "
          f"{r['plays_per_game']:7.2f} {pr:>9s}")
