"""
enemies.py — 末日废土+赛博克系 敌人定义
6 个敌人:5 普通 + 1 Boss
意图(intent)系统:每个回合敌人会预告下一步行动
"""
import random

# ---------- 意图类型 ----------
# attack: 普通攻击
# attack_buff: 攻击+给自己加力量
# defend: 加护甲
# debuff: 给玩家加 debuff(腐蚀/SAN扣除等)
# big_attack: 大招(回合数较长才出一次)

INTENT_TYPES = ["attack", "attack_buff", "defend", "debuff", "big_attack"]


# ---------- 敌人模板 ----------
ENEMIES = {
    # ===== 普通敌人 =====
    0: {
        "name": "锈蚀蠕虫",
        "hp": 20,
        "intents": [
            {"type": "attack", "value": 5, "desc": "啃咬 5"},
            {"type": "attack", "value": 7, "desc": "酸液喷射 7"},
            {"type": "defend", "value": 4, "desc": "蜷缩 +4 护甲"},
        ],
        "intent_weights": [0.5, 0.3, 0.2],
        "tier": "easy",
    },
    1: {
        "name": "拾荒帮匪",
        "hp": 30,
        "intents": [
            {"type": "attack", "value": 6, "desc": "生锈管道 6"},
            {"type": "attack_buff", "value": 4, "buff": 2, "desc": "怒吼 +2力量,攻击4"},
            {"type": "defend", "value": 6, "desc": "破盾 +6 护甲"},
        ],
        "intent_weights": [0.5, 0.25, 0.25],
        "tier": "easy",
    },
    2: {
        "name": "数据巫师",
        "hp": 25,
        "intents": [
            {"type": "attack", "value": 8, "desc": "数据脉冲 8"},
            {"type": "debuff", "value": 2, "debuff": "corrosion", "desc": "腐蚀代码 +2腐蚀"},
            {"type": "debuff", "value": 3, "debuff": "san_loss", "desc": "心智入侵 -3 SAN"},
        ],
        "intent_weights": [0.4, 0.3, 0.3],
        "tier": "medium",
    },
    3: {
        "name": "白噪同化体",
        "hp": 45,
        "intents": [
            {"type": "attack", "value": 10, "desc": "同化触手 10"},
            {"type": "defend", "value": 8, "desc": "白噪屏障 +8 护甲"},
            {"type": "big_attack", "value": 16, "cooldown": 3, "desc": "[蓄力中] 同化爆发 16"},
        ],
        "intent_weights": [0.55, 0.3, 0.15],
        "tier": "hard",
    },
    4: {
        "name": "暗网影子",
        "hp": 25,
        "intents": [
            {"type": "attack", "value": 4, "desc": "影刃 4 (×2次)", "hits": 2},
            {"type": "debuff", "value": 1, "debuff": "weak", "desc": "暗影笼罩 +1虚弱"},
            {"type": "defend", "value": 5, "desc": "潜行 +5 护甲"},
        ],
        "intent_weights": [0.5, 0.3, 0.2],
        "tier": "medium",
    },
    # ===== Boss =====
    5: {
        "name": "白噪本身",
        "hp": 65,
        "intents": [
            {"type": "attack", "value": 12, "desc": "现实撕裂 12"},
            {"type": "attack_buff", "value": 8, "buff": 3, "desc": "低语 +3力量,攻击8"},
            {"type": "debuff", "value": 4, "debuff": "san_loss", "desc": "认知污染 -4 SAN"},
            {"type": "big_attack", "value": 22, "cooldown": 4, "desc": "[蓄力中] 万物归噪 22"},
        ],
        "intent_weights": [0.4, 0.25, 0.2, 0.15],
        "tier": "boss",
    },
}

NUM_ENEMIES = len(ENEMIES)


# ---------- 工厂 / 状态 ----------
def make_enemy(enemy_id, rng=None):
    """根据 ID 创建一个敌人状态实例"""
    if rng is None:
        rng = random.Random()
    template = ENEMIES[enemy_id]
    return {
        "id": enemy_id,
        "name": template["name"],
        "hp": template["hp"],
        "max_hp": template["hp"],
        "armor": 0,
        "strength": 0,        # 攻击加成
        "weak": 0,            # 自身被虚弱(攻击-25%),回合数
        "next_intent": None,
        "big_attack_cd": 0,   # 大招冷却
        "_rng": rng,
        "_template": template,
    }


def roll_intent(enemy):
    """为敌人决定下一回合意图"""
    template = enemy["_template"]
    rng = enemy["_rng"]
    intents = template["intents"]
    weights = list(template["intent_weights"])

    # 大招冷却:在 cooldown 没到之前不出大招
    for i, it in enumerate(intents):
        if it["type"] == "big_attack" and enemy["big_attack_cd"] > 0:
            weights[i] = 0.0

    # 归一化(全部为 0 则强制选第一个)
    total = sum(weights)
    if total <= 0:
        chosen = intents[0]
    else:
        weights = [w / total for w in weights]
        chosen = rng.choices(intents, weights=weights, k=1)[0]

    if chosen["type"] == "big_attack":
        enemy["big_attack_cd"] = chosen.get("cooldown", 3)
    else:
        enemy["big_attack_cd"] = max(0, enemy["big_attack_cd"] - 1)

    enemy["next_intent"] = chosen
    return chosen


def get_enemy_pool(stage):
    """根据楼层阶段返回可用的敌人 ID 池
    stage 0: easy 楼层 (前 1-3 层)
    stage 1: medium 楼层 (4-6 层)
    stage 2: hard 楼层 (7-9 层)
    stage 3: boss 楼层 (10 层)
    """
    if stage == 0:
        return [0, 1]
    elif stage == 1:
        return [1, 2, 4]
    elif stage == 2:
        return [2, 3, 4]
    elif stage == 3:
        return [5]
    else:
        return [0, 1, 2, 3, 4]
