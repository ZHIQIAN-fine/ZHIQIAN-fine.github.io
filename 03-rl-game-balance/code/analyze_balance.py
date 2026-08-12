"""
Mini Whisper - Balance & Strategy Analysis (1000 episodes)
Outputs:
  results/runs.csv          per-episode records
  results/summary.json      aggregate stats
  results/fig1_reward.png   reward distribution
  results/fig2_floor.png    floor reach rate + death distribution
  results/fig3_curve.png    HP/SAN per-floor average
  results/fig4_cards.png    card usage frequency (top)
"""
import os, sys, json, time
sys.path.insert(0, "/content/drive/MyDrive/mini_whisper")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Use CJK font ONLY for tick labels that contain card names; chart text stays English.
CJK_PATH = None
for p in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
        CJK_PATH = p
        plt.rcParams["font.sans-serif"] = [
            fm.FontProperties(fname=p).get_name(), "DejaVu Sans"
        ]
        break
plt.rcParams["axes.unicode_minus"] = False

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env import MiniWhisperEnv
from cards import CARDS

MODEL_PATH = "/content/drive/MyDrive/mini_whisper/training_output/maskable_ppo_final.zip"
OUT_DIR    = "/content/drive/MyDrive/mini_whisper/results"
N_RUNS     = 1000
END_TURN   = 15
MAX_FLOOR  = 10

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[load] {MODEL_PATH}")
model = MaskablePPO.load(MODEL_PATH)

# ========== Run N episodes ==========
records = []
card_use = {}
hp_traj  = {f: [] for f in range(1, MAX_FLOOR + 1)}
san_traj = {f: [] for f in range(1, MAX_FLOOR + 1)}

t0 = time.time()
for i in range(N_RUNS):
    env = MiniWhisperEnv(seed=20000 + i)
    env_w = ActionMasker(env, lambda x: x.unwrapped.action_masks())
    obs, _ = env_w.reset()
    done = False
    total_r = 0.0
    turns = 0
    cards_played = 0
    last_floor_seen = 0
    while not done:
        mask = env_w.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)

        cf = env.floor
        if cf != last_floor_seen and cf >= 1:
            hp_traj[cf].append(env.player["hp"])
            san_traj[cf].append(env.player["san"])
            last_floor_seen = cf

        if action != END_TURN:
            hand = env.player.get("hand", [])
            if action < len(hand):
                cid = hand[action]
                card_use[cid] = card_use.get(cid, 0) + 1
                cards_played += 1

        obs, r, term, trunc, info = env_w.step(action)
        total_r += r
        if action == END_TURN:
            turns += 1
        done = term or trunc

    records.append({
        "ep": i,
        "seed": 20000 + i,
        "reward": total_r,
        "floor_reached": env.floor,
        "victory": bool(getattr(env, "victory", False)),
        "turns": turns,
        "cards_played": cards_played,
        "final_hp": env.player["hp"],
        "final_san": env.player["san"],
    })

    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{N_RUNS}] elapsed={elapsed:.1f}s  eta={elapsed/(i+1)*(N_RUNS-i-1):.1f}s")

dt = time.time() - t0
print(f"[done] {N_RUNS} runs in {dt:.1f}s ({dt/N_RUNS*1000:.1f} ms/run)")

# ========== Aggregate ==========
df = pd.DataFrame(records)
df.to_csv(f"{OUT_DIR}/runs.csv", index=False)

summary = {
    "n_runs":              N_RUNS,
    "mean_reward":         float(df.reward.mean()),
    "std_reward":          float(df.reward.std()),
    "max_reward":          float(df.reward.max()),
    "min_reward":          float(df.reward.min()),
    "mean_floor":          float(df.floor_reached.mean()),
    "boss_reach_rate":     float((df.floor_reached >= 10).mean()),
    "win_rate":            float(df.victory.mean()),
    "mean_turns":          float(df.turns.mean()),
    "mean_cards_per_run":  float(df.cards_played.mean()),
}
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n=== Aggregate Statistics ===")
for k, v in summary.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

# ========== Figure 1: Reward Distribution ==========
fig, ax = plt.subplots(figsize=(10, 5))
wins   = df[df.victory].reward
losses = df[~df.victory].reward
ax.hist([losses, wins], bins=30, stacked=True,
        color=["#dc4646", "#5ac882"], label=["Defeat", "Victory"], edgecolor="white")
ax.axvline(df.reward.mean(), color="#50c8e6", lw=2, ls="--",
           label=f"Mean = {df.reward.mean():.1f}")
ax.set_xlabel("Total Reward")
ax.set_ylabel("Episodes")
ax.set_title(f"Reward Distribution over {N_RUNS} Episodes (MaskablePPO 300k)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_reward.png", dpi=150)
plt.close()
print("[fig1] saved")

# ========== Figure 2: Floor Reach + Death ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

reach = [(df.floor_reached >= f).mean() * 100 for f in range(1, MAX_FLOOR + 1)]
ax = axes[0]
bars = ax.bar(range(1, MAX_FLOOR + 1), reach,
              color="#50c8e6", edgecolor="#1a1828")
ax.set_xlabel("Floor")
ax.set_ylabel("Reach Rate (%)")
ax.set_title("Per-Floor Reach Rate")
ax.set_ylim(0, 105)
ax.set_xticks(range(1, MAX_FLOOR + 1))
ax.grid(alpha=0.3, axis="y")
for b, v in zip(bars, reach):
    ax.text(b.get_x() + b.get_width()/2, v + 1,
            f"{v:.0f}%", ha="center", fontsize=9)

dead = df[~df.victory]
ax = axes[1]
death_counts = dead.floor_reached.value_counts().sort_index()
ax.bar(death_counts.index, death_counts.values,
       color="#dc4646", edgecolor="#1a1828")
ax.set_xlabel("Floor of Death")
ax.set_ylabel("Number of Defeats")
ax.set_title(f"Death-Floor Distribution ({len(dead)} losses out of {N_RUNS})")
ax.set_xticks(range(1, MAX_FLOOR + 1))
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_floor.png", dpi=150)
plt.close()
print("[fig2] saved")

# ========== Figure 3: HP / SAN per Floor ==========
fig, ax = plt.subplots(figsize=(10, 5))
floors = list(range(1, MAX_FLOOR + 1))
hp_mean  = [np.mean(hp_traj[f])  if hp_traj[f]  else np.nan for f in floors]
hp_std   = [np.std (hp_traj[f])  if hp_traj[f]  else 0      for f in floors]
san_mean = [np.mean(san_traj[f]) if san_traj[f] else np.nan for f in floors]
san_std  = [np.std (san_traj[f]) if san_traj[f] else 0      for f in floors]

ax.plot(floors, hp_mean,  "o-", color="#dc4646", lw=2, label="HP (mean)")
ax.fill_between(floors,
                [m-s for m,s in zip(hp_mean,hp_std)],
                [m+s for m,s in zip(hp_mean,hp_std)],
                color="#dc4646", alpha=0.15, label="HP ±1σ")
ax.plot(floors, san_mean, "s-", color="#b45ad8", lw=2, label="SAN (mean)")
ax.fill_between(floors,
                [m-s for m,s in zip(san_mean,san_std)],
                [m+s for m,s in zip(san_mean,san_std)],
                color="#b45ad8", alpha=0.15, label="SAN ±1σ")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Floor")
ax.set_ylabel("Resource Value")
ax.set_title("Average Player HP / SAN per Floor (mean ±1σ)")
ax.set_xticks(floors)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_curve.png", dpi=150)
plt.close()
print("[fig3] saved")

# ========== Figure 4: Card Usage Frequency ==========
fig, ax = plt.subplots(figsize=(11, 5.5))
items = sorted(card_use.items(), key=lambda x: -x[1])
# Card names stay in CJK (game data); axis title and percentage labels are English.
labels_cn = [CARDS.get(cid, {}).get("name", f"Card{cid}") for cid, _ in items]
counts    = [c for _, c in items]
total     = sum(counts) if items else 1
pct       = [c/total*100 for c in counts]

bars = ax.barh(labels_cn[::-1], counts[::-1],
               color="#50c8e6", edgecolor="#1a1828")
ax.set_xlabel("Total Plays")
ax.set_title(f"Card Usage Frequency over {N_RUNS} Episodes "
             f"(total plays = {total})")
for b, c, p in zip(bars, counts[::-1], pct[::-1]):
    ax.text(b.get_width() + max(counts) * 0.005,
            b.get_y() + b.get_height()/2,
            f"{c}  ({p:.1f}%)", va="center", fontsize=9)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_cards.png", dpi=150)
plt.close()
print("[fig4] saved")

print(f"\n[all done] outputs in {OUT_DIR}")
print(f"  - runs.csv         {N_RUNS} rows")
print(f"  - summary.json")
print(f"  - fig1_reward.png  Reward distribution")
print(f"  - fig2_floor.png   Floor reach + death distribution")
print(f"  - fig3_curve.png   HP/SAN per-floor curves")
print(f"  - fig4_cards.png   Card usage frequency")
