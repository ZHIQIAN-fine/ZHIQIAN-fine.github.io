"""
Mini Whisper - 录制 AI 对战演示视频 (无头渲染 + 中文字体)
"""
import os, sys, subprocess, shutil, glob
sys.path.insert(0, "/content/drive/MyDrive/mini_whisper")

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1280, 720))
display.start()

import pygame
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env import MiniWhisperEnv
from cards import CARDS

# ========== 配置 ==========
W, H        = 1280, 720
FPS         = 30
FRAMES_PER_ACTION = 36
MAX_FRAMES  = 90 * FPS
END_TURN_ACTION = 15

OUT_DIR    = "/content/work/video_frames"
OUT_MP4    = "/content/work/demo.mp4"
DRIVE_MP4  = "/content/drive/MyDrive/mini_whisper/demo.mp4"
MODEL_PATH = "/content/drive/MyDrive/mini_whisper/training_output/maskable_ppo_final.zip"
SEED       = 42

if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ========== pygame + 中文字体 ==========
pygame.init()
screen = pygame.Surface((W, H))

def find_cjk_font():
    """显式找一个支持中文的字体文件路径"""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for p in candidates:
        if os.path.exists(p): return p
    # 兜底:扫描所有 noto cjk
    found = glob.glob("/usr/share/fonts/**/*CJK*.ttc", recursive=True) + \
            glob.glob("/usr/share/fonts/**/wqy*.ttc",  recursive=True)
    return found[0] if found else None

CJK_FONT = find_cjk_font()
print(f"[font] CJK font = {CJK_FONT}")

def load_font(size, bold=False):
    if CJK_FONT:
        f = pygame.font.Font(CJK_FONT, size)
        f.set_bold(bold)
        return f
    return pygame.font.SysFont("DejaVuSans", size, bold=bold)

font_l = load_font(34, bold=True)
font_m = load_font(22)
font_s = load_font(17)

# ========== 配色 ==========
BG, PANEL = (16,14,24), (28,25,42)
FG, DIM   = (220,215,200), (110,105,95)
RED, GREEN, CYAN, PURPLE, YELLOW = (220,70,70),(90,200,130),(80,200,230),(180,90,220),(240,200,80)

def text(s, font, color, x, y, center=False):
    surf = font.render(str(s), True, color)
    r = surf.get_rect()
    if center: r.center = (x, y)
    else:      r.topleft = (x, y)
    screen.blit(surf, r)

def bar(x, y, w, h, val, vmax, color, label):
    pygame.draw.rect(screen, DIM, (x, y, w, h), 2)
    if vmax > 0:
        fill = max(0, min(w-4, int((w-4)*val/vmax)))
        pygame.draw.rect(screen, color, (x+2, y+2, fill, h-4))
    text(f"{label}: {val}/{vmax}", font_s, FG, x, y - 22)

def draw_frame(env, action_label="", reward_so_far=0.0, banner=None, banner_color=None):
    screen.fill(BG)
    p = env.player; e = env.enemy
    text("MINI WHISPER  —  末日废土 × 赛博克系", font_l, CYAN, W//2, 30, center=True)
    text(f"AI智能体 (MaskablePPO 300k) | 第 {env.floor}/{env.max_floors} 层 | 回合 {env.turn} | 奖励 {reward_so_far:+.1f}",
         font_m, DIM, W//2, 65, center=True)

    # 玩家
    pygame.draw.rect(screen, PANEL, (40, 100, 380, 280))
    text("玩家", font_m, YELLOW, 60, 115)
    bar(60, 165, 320, 26, p["hp"],  p["max_hp"],  RED,    "HP")
    bar(60, 225, 320, 26, p["san"], p["max_san"], PURPLE, "SAN")
    text(f"能量: {p['energy']}/{p['max_energy']}", font_m, CYAN, 60, 270)
    text(f"护甲: {p['armor']}",                     font_m, FG,   60, 300)
    text(f"力量 {p['strength']}  虚弱 {p['weak']}  腐蚀 {p['corrosion']}",
         font_s, DIM, 60, 335)

    # 敌人
    pygame.draw.rect(screen, PANEL, (440, 100, 420, 280))
    text("敌人", font_m, YELLOW, 460, 115)
    if e:
        text(e["name"], font_m, RED, 460, 145)
        bar(460, 195, 380, 26, e["hp"], e["max_hp"], RED, "HP")
        intent = e.get("next_intent", {})
        intent_desc = intent.get("desc", "?") if isinstance(intent, dict) else str(intent)
        text(f"意图: {intent_desc}", font_s, YELLOW, 460, 240)
        text(f"护甲 {e.get('armor',0)}  力量 {e.get('strength',0)}  虚弱 {e.get('weak',0)}",
             font_s, DIM, 460, 268)

    # 规则
    pygame.draw.rect(screen, PANEL, (880, 100, 360, 280))
    text("当前规则", font_m, YELLOW, 900, 115)
    rules = getattr(env, "active_rules", []) or []
    if rules:
        for i, rid in enumerate(rules):
            text(f"• 规则 [{rid}]", font_m, PURPLE, 900, 150 + i*32)
    else:
        text("(无)", font_s, DIM, 900, 150)

    # 手牌
    text("手牌", font_m, YELLOW, 60, 400)
    hand = p.get("hand", []) or []
    card_w, card_h = 220, 270
    gap = 20
    total_w = len(hand) * card_w + (len(hand)-1) * gap if hand else 0
    start_x = (W - total_w) // 2 if hand else 60
    for i, cid in enumerate(hand):
        c = CARDS.get(cid, {})
        cx = start_x + i * (card_w + gap)
        cy = 430
        is_played = (action_label.startswith("出牌") and f"槽 {i}]" in action_label)
        border_color = GREEN if is_played else CYAN
        pygame.draw.rect(screen, PANEL, (cx, cy, card_w, card_h))
        pygame.draw.rect(screen, border_color, (cx, cy, card_w, card_h), 3)
        text(c.get("name", f"卡{cid}"),       font_m, FG,     cx + 12, cy + 10)
        text(f"消耗: {c.get('cost', 0)}",     font_s, YELLOW, cx + 12, cy + 45)
        text(f"类型: {c.get('type','?')}",    font_s, CYAN,   cx + 12, cy + 68)
        text(f"槽位 [{i}]",                   font_s, DIM,    cx + 12, cy + card_h - 28)

    # 动作条
    pygame.draw.rect(screen, (40, 50, 70), (0, H - 55, W, 55))
    if action_label:
        text(f"AI 动作  ▶  {action_label}", font_m, GREEN, W//2, H - 28, center=True)

    # 横幅
    if banner:
        bw, bh = 600, 120
        bx, by = (W-bw)//2, (H-bh)//2
        pygame.draw.rect(screen, (0,0,0), (bx, by, bw, bh))
        pygame.draw.rect(screen, banner_color or YELLOW, (bx, by, bw, bh), 4)
        text(banner, font_l, banner_color or YELLOW, W//2, H//2, center=True)

def save_frame(idx):
    pygame.image.save(screen, f"{OUT_DIR}/frame_{idx:05d}.png")

# ========== 加载模型 ==========
print("[load] loading model ...")
model = MaskablePPO.load(MODEL_PATH)

env = MiniWhisperEnv(seed=SEED)
env_w = ActionMasker(env, lambda x: x.unwrapped.action_masks())
obs, _ = env_w.reset()

frame_idx = 0; total_r = 0.0; done = False

print("[record] starting ...")
for _ in range(int(FPS * 1.5)):
    draw_frame(env, "对局开始", total_r, banner="对局开始", banner_color=CYAN)
    save_frame(frame_idx); frame_idx += 1

while not done and frame_idx < MAX_FRAMES:
    mask = env_w.action_masks()
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    action = int(action)
    if action == END_TURN_ACTION:
        label = "结束回合"
    else:
        hand = env.player.get("hand", [])
        if action < len(hand):
            cname = CARDS.get(hand[action], {}).get("name", f"卡{hand[action]}")
            label = f"出牌 [{cname}] [槽 {action}]"
        else:
            label = f"槽 {action}"

    obs, r, term, trunc, info = env_w.step(action)
    total_r += r; done = term or trunc

    for _ in range(FRAMES_PER_ACTION):
        if frame_idx >= MAX_FRAMES: break
        draw_frame(env, label, total_r); save_frame(frame_idx); frame_idx += 1

victory   = bool(getattr(env, "victory", False))
end_msg   = "胜利!" if victory else "失败"
end_color = GREEN if victory else RED
for _ in range(FPS * 2):
    if frame_idx >= MAX_FRAMES: break
    draw_frame(env, end_msg, total_r, banner=end_msg, banner_color=end_color)
    save_frame(frame_idx); frame_idx += 1

print(f"[record] {frame_idx} frames ({frame_idx/FPS:.1f}s)  victory={victory}  reward={total_r:+.2f}")

print("[ffmpeg] encoding ...")
subprocess.run([
    "ffmpeg","-y","-loglevel","error",
    "-framerate", str(FPS),
    "-i", f"{OUT_DIR}/frame_%05d.png",
    "-c:v","libx264","-pix_fmt","yuv420p","-crf","20",
    "-vf","scale=1280:720",
    OUT_MP4,
], check=True)
print(f"[ffmpeg] saved {OUT_MP4}  ({os.path.getsize(OUT_MP4)/1024/1024:.2f} MB)")

shutil.copy(OUT_MP4, DRIVE_MP4)
print(f"[sync] {DRIVE_MP4}")
display.stop()
print("[done]")
