"""命令行人工试玩 — 用于 sanity check 整个环境是否好玩 / 平衡是否合理"""
from env import MiniWhisperEnv, ACTION_END_TURN
from cards import CARDS

def main():
    env = MiniWhisperEnv(render_mode="ansi", seed=None)
    obs, info = env.reset()
    print(env.render())
    done = False
    total_reward = 0.0
    while not done:
        mask = env.action_masks()
        legal = [i for i, m in enumerate(mask) if m]
        print(f"\n合法动作: " + ", ".join(
            f"{i}={'结束回合' if i == ACTION_END_TURN else CARDS[i]['name']}"
            for i in legal
        ))
        try:
            raw = input("> 你的选择(数字): ").strip()
            if raw == "q":
                break
            a = int(raw)
        except (ValueError, EOFError):
            print("无效输入,默认结束回合")
            a = ACTION_END_TURN
        obs, r, done, trunc, info = env.step(a)
        total_reward += r
        print(f"\nreward = {r:+.2f}  累计 = {total_reward:+.2f}")
        print(env.render())
    print(f"\n=== 游戏结束 胜利={info.get('victory')} 累计 reward = {total_reward:+.2f} ===")

if __name__ == "__main__":
    main()
