"""rules.py v2 — 平衡化版本
- RULES[2] 引入 free_cards=2 免罚阈值
- RULES[1] 抽牌惩罚改为只对超过 5 张的部分扣(避免每回合必扣)
- RULES[3] 攻击牌惩罚保留(novelty 体现)
- RULES[0/4] 维持
"""
import random

RULES = {
    0: {
        "name": "禁止使用「不该看的日志」",
        "desc": "本层禁用 [不该看的日志]。违反 -3 SAN。",
        "trigger": "on_play_card",
        "card_id_blacklist": [9],
        "penalty": 3,
    },
    1: {
        "name": "抽牌过多扣 SAN",
        "desc": "本层每回合抽超过 5 张牌,每多抽 1 张 -1 SAN。",
        "trigger": "on_draw",
        "penalty": 1,
        "free_draws": 5,        # ⭐ 阈值
    },
    2: {
        "name": "手牌过多扣 SAN",
        "desc": "回合结束时手牌数 >2,每超出 1 张 -2 SAN。",
        "trigger": "on_turn_end",
        "penalty_per_card": 2,
        "free_cards": 2,        # ⭐ 阈值
    },
    3: {
        "name": "禁止使用攻击牌",
        "desc": "本层禁止打出攻击类卡牌。违反 -2 SAN。",
        "trigger": "on_play_card",
        "card_type_blacklist": ["attack"],
        "penalty": 2,
    },
    4: {
        "name": "必须打出技能/能力牌",
        "desc": "回合内未打出技能或能力牌则 -3 SAN。",
        "trigger": "on_turn_end",
        "require_card_type": ["skill", "power"],
        "penalty": 3,
    },
}

NUM_RULES = len(RULES)


def roll_rules_for_stage(stage, rng=None):
    if rng is None:
        rng = random.Random()
    n = 1
    return rng.sample(list(RULES.keys()), k=min(n, NUM_RULES))


def check_play_card_violation(active_rules, card, player):
    penalty = 0
    multiplier = player.get("rule_penalty_multiplier", 1)
    for rid in active_rules:
        rule = RULES[rid]
        if rule["trigger"] != "on_play_card":
            continue
        violated = False
        if "card_id_blacklist" in rule and card.get("id") in rule["card_id_blacklist"]:
            violated = True
        if "card_type_blacklist" in rule and card.get("type") in rule["card_type_blacklist"]:
            violated = True
        if violated:
            penalty += rule["penalty"] * multiplier
    return penalty


def check_draw_violation(active_rules, num_drawn, player):
    """抽牌违反:超过 free_draws 才扣"""
    penalty = 0
    multiplier = player.get("rule_penalty_multiplier", 1)
    for rid in active_rules:
        rule = RULES[rid]
        if rule["trigger"] != "on_draw":
            continue
        free = rule.get("free_draws", 0)
        over = max(0, num_drawn - free)
        penalty += rule["penalty"] * over * multiplier
    return penalty


def check_turn_end_violation(active_rules, player, played_cards_this_turn):
    penalty = 0
    multiplier = player.get("rule_penalty_multiplier", 1)
    hand_size = len(player.get("hand", []))
    for rid in active_rules:
        rule = RULES[rid]
        if rule["trigger"] != "on_turn_end":
            continue
        if "penalty_per_card" in rule:
            free = rule.get("free_cards", 0)        # ⭐ 阈值
            over = max(0, hand_size - free)
            penalty += rule["penalty_per_card"] * over * multiplier
        if "require_card_type" in rule:
            required = set(rule["require_card_type"])
            played_types = {c.get("type") for c in played_cards_this_turn}
            if not (required & played_types):
                penalty += rule["penalty"] * multiplier
    return penalty


def get_rule_descriptions(active_rules):
    return [RULES[rid]["desc"] for rid in active_rules]
