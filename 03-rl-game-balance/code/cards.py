"""
cards.py — 末日废土+赛博克系 卡牌定义(v2 重写版)
设计原则:
- 所有 effect 统一签名:effect(ctx) -> dict
- ctx 是 dict,包含 player/enemy/rng/deck/discard/exhaust/hand
- effect 返回标准状态变化 dict(env 用于 reward shaping & log)

卡牌总数:15
"""
import random


# ============================================================
#  通用工具
# ============================================================
def _empty_result():
    return {
        "damage_dealt": 0,
        "armor_gained": 0,
        "heal": 0,
        "san_gain": 0,
        "extra": {},
    }


def _deal_damage(ctx, base):
    """统一伤害函数:考虑玩家 OVERCLOCK buff、力量、敌人虚弱、护甲。
    返回实际穿透伤害(护甲外的)+ 护甲消耗(总有效伤害)。"""
    p = ctx["player"]
    e = ctx["enemy"]

    # 玩家加成
    bonus = p.get("overclock_buff", 0) + p.get("strength", 0)
    if p.get("weak", 0) > 0:
        # 虚弱:玩家攻击 -25%
        total = int((base + bonus) * 0.75)
    else:
        total = base + bonus

    # 敌人护甲先吸
    if e.get("armor", 0) > 0:
        absorbed = min(e["armor"], total)
        e["armor"] -= absorbed
        total -= absorbed
        damage_dealt = absorbed   # 护甲损失也计入"造成伤害"
    else:
        damage_dealt = 0

    # 剩余打 HP
    if total > 0:
        e["hp"] -= total
        damage_dealt += total

    return damage_dealt


def _gain_armor(ctx, amount):
    p = ctx["player"]
    p["armor"] += amount
    return amount


def _heal(ctx, amount):
    p = ctx["player"]
    before = p["hp"]
    p["hp"] = min(p["max_hp"], p["hp"] + amount)
    return p["hp"] - before


def _heal_san(ctx, amount):
    p = ctx["player"]
    before = p["san"]
    p["san"] = min(p["max_san"], p["san"] + amount)
    return p["san"] - before


# ============================================================
#  卡牌效果函数(全部 effect(ctx) -> dict)
# ============================================================

# ---- 0:生锈匕首(基础攻击)----
def _rusty_dagger(ctx):
    r = _empty_result()
    r["damage_dealt"] = _deal_damage(ctx, 6)
    return r

# ---- 1:过载电浆刃(高伤,自伤)----
def _plasma_blade(ctx):
    r = _empty_result()
    r["damage_dealt"] = _deal_damage(ctx, 14)
    # 自伤 2(穿透自身护甲)
    p = ctx["player"]
    self_dmg = 2
    absorb = min(p.get("armor", 0), self_dmg)
    p["armor"] -= absorb
    p["hp"] -= (self_dmg - absorb)
    r["extra"]["self_damage"] = self_dmg
    return r

# ---- 2:触发链×2(2 次小攻击)----
def _trigger_chain(ctx):
    r = _empty_result()
    total = 0
    for _ in range(2):
        total += _deal_damage(ctx, 4)
    r["damage_dealt"] = total
    return r

# ---- 3:离子护盾(基础防御)----
def _ion_shield(ctx):
    r = _empty_result()
    r["armor_gained"] = _gain_armor(ctx, 5)
    return r

# ---- 4:冥想程序(回血+回SAN)----
def _meditation(ctx):
    r = _empty_result()
    r["heal"] = _heal(ctx, 6)
    r["san_gain"] = _heal_san(ctx, 1)
    return r

# ---- 5:404·死灵协议(中等攻击)----
def _necro_404(ctx):
    r = _empty_result()
    r["damage_dealt"] = _deal_damage(ctx, 9)
    return r

# ---- 6:进程冻结(虚弱敌人)----
def _process_freeze(ctx):
    r = _empty_result()
    e = ctx["enemy"]
    e["weak"] = e.get("weak", 0) + 2
    r["extra"]["enemy_weak"] = 2
    return r

# ---- 7:代码污染(给敌人加腐蚀)----
def _code_corrupt(ctx):
    r = _empty_result()
    e = ctx["enemy"]
    # 简化:直接打 5 伤害(代表持续腐蚀)
    r["damage_dealt"] = _deal_damage(ctx, 5)
    return r

# ---- 8:OVERCLOCK(自我增益,本回合后续攻击 +3)----
def _overclock(ctx):
    r = _empty_result()
    p = ctx["player"]
    p["overclock_buff"] = p.get("overclock_buff", 0) + 3
    r["extra"]["overclock_added"] = 3
    return r

# ---- 9:不该看的日志(M1 - 预见)----
def _forbidden_log(ctx):
    r = _empty_result()
    p = ctx["player"]
    p["foresight_turns"] = p.get("foresight_turns", 0) + 2
    r["extra"]["foresight_turns_added"] = 2
    # 代价:扣 1 SAN
    p["san"] -= 1
    r["extra"]["san_cost"] = 1
    return r

# ---- 10:拆解(M2 - 高伤但消耗一张随机手牌)----
def _dismantle(ctx):
    r = _empty_result()
    p = ctx["player"]
    rng = ctx.get("rng") or random
    # 消耗一张随机手牌(if 有的话)
    sacrificed = None
    if p["hand"]:
        sacrificed = rng.choice(p["hand"])
        p["hand"].remove(sacrificed)
        p["exhaust"].append(sacrificed)
    # 高伤
    r["damage_dealt"] = _deal_damage(ctx, 10)
    if sacrificed is not None:
        r["extra"]["sacrificed_card"] = sacrificed
    return r

# ---- 11:格式化记忆(M3 - 把一张随机牌从牌组永久移除)----
def _format_memory(ctx):
    r = _empty_result()
    p = ctx["player"]
    rng = ctx.get("rng") or random
    # 优先从弃牌堆移除,没有则从牌组,再没有则从手牌
    pool_name, target = None, None
    for name in ("discard", "deck", "hand"):
        if p[name]:
            pool_name = name
            target = rng.choice(p[name])
            break
    if pool_name is not None:
        p[pool_name].remove(target)
        # 不进 exhaust,直接消失
        r["extra"]["formatted_card"] = target
        r["extra"]["formatted_from"] = pool_name
    return r

# ---- 12:亵渎协议(M5 - 本回合规则惩罚×2,但伤害+8)----
def _profane_protocol(ctx):
    r = _empty_result()
    p = ctx["player"]
    p["rule_penalty_multiplier"] = 2
    r["damage_dealt"] = _deal_damage(ctx, 8)
    r["extra"]["rule_penalty_x2"] = True
    return r

# ---- 13:无名之吻(M7 疯狂 - SAN<=10 才出现)----
def _nameless_kiss(ctx):
    r = _empty_result()
    # 巨额伤害 + 强制扣 SAN
    r["damage_dealt"] = _deal_damage(ctx, 20)
    p = ctx["player"]
    p["san"] -= 3
    r["extra"]["madness_san_cost"] = 3
    return r

# ---- 14:黑色低语(M7 疯狂 - SAN<=10 才出现)----
def _black_whisper(ctx):
    r = _empty_result()
    p = ctx["player"]
    e = ctx["enemy"]
    # 给敌人加 3 虚弱 + 自己回 5 SAN
    e["weak"] = e.get("weak", 0) + 3
    r["san_gain"] = _heal_san(ctx, 5)
    r["extra"]["enemy_weak"] = 3
    return r


# ============================================================
#  CARDS 注册表
# ============================================================
CARDS = {
    0:  {"name": "生锈匕首",      "cost": 1, "type": "attack", "effect": _rusty_dagger,    "dest": "discard"},
    1:  {"name": "过载电浆刃",    "cost": 2, "type": "attack", "effect": _plasma_blade,    "dest": "discard"},
    2:  {"name": "触发链×2",      "cost": 1, "type": "attack", "effect": _trigger_chain,   "dest": "discard"},
    3:  {"name": "离子护盾",      "cost": 1, "type": "skill",  "effect": _ion_shield,      "dest": "discard"},
    4:  {"name": "冥想程序",      "cost": 1, "type": "skill",  "effect": _meditation,      "dest": "discard"},
    5:  {"name": "404·死灵协议",  "cost": 2, "type": "attack", "effect": _necro_404,       "dest": "discard"},
    6:  {"name": "进程冻结",      "cost": 1, "type": "skill",  "effect": _process_freeze,  "dest": "discard"},
    7:  {"name": "代码污染",      "cost": 1, "type": "attack", "effect": _code_corrupt,    "dest": "discard"},
    8:  {"name": "OVERCLOCK",     "cost": 2, "type": "power",  "effect": _overclock,       "dest": "exhaust"},
    9:  {"name": "不该看的日志",  "cost": 0, "type": "skill",  "effect": _forbidden_log,   "dest": "exhaust"},
    10: {"name": "拆解",          "cost": 0, "type": "attack", "effect": _dismantle,       "dest": "exhaust"},
    11: {"name": "格式化记忆",    "cost": 1, "type": "skill",  "effect": _format_memory,   "dest": "exhaust", "loot_only": True},
    12: {"name": "亵渎协议",      "cost": 2, "type": "skill",  "effect": _profane_protocol,"dest": "exhaust", "loot_only": True},
    13: {"name": "无名之吻",      "cost": 1, "type": "attack", "effect": _nameless_kiss,   "dest": "exhaust", "madness": True},
    14: {"name": "黑色低语",      "cost": 0, "type": "skill",  "effect": _black_whisper,   "dest": "exhaust", "madness": True},
}

NUM_CARDS = len(CARDS)


# ============================================================
#  辅助函数
# ============================================================
def get_starter_cards():
    """初始牌组:5 生锈匕首 + 5 离子护盾"""
    return [0] * 5 + [3] * 5


def get_loot_pool():
    """战斗胜利后随机掉落的卡(loot only 也包括)"""
    return [
        cid for cid, c in CARDS.items()
        if not c.get("madness", False)        # 疯狂卡不能 loot
    ]


def get_madness_cards():
    """SAN <= 10 时可能出现的疯狂卡"""
    return [cid for cid, c in CARDS.items() if c.get("madness", False)]
