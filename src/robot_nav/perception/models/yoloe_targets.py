"""常用目标的固定文本提示词与场景线索；不加载 PF 模型。"""

import json
from pathlib import Path


def text_prompts(class_texts):
    """只编码本轮目标；明确同义词使用固定名称，未收录的目标保留原文。"""
    catalog = json.loads(Path(__file__).with_name("yoloe_targets.json").read_text(encoding="utf-8"))
    vocabulary = {name.casefold(): name for name in catalog["vocabulary"]}
    aliases = {alias.casefold(): entry["text_prompt"]
               for entry in catalog["targets"] for alias in entry["aliases"]}
    prompts = []
    for target in class_texts:
        key = " ".join(target.split()).casefold()
        prompts.append(aliases.get(key, vocabulary.get(key, target)))
    # 清单若包含两个同义类别，不把同一个提示词绑定到不同的任务身份。
    duplicates = {name for name in prompts if prompts.count(name) > 1}
    return tuple(target if prompt in duplicates else prompt
                 for target, prompt in zip(class_texts, prompts))


def predefined_scene_objects(scene):
    """完整场景名称命中时使用预生成清单；带额外限定的目标仍交给 Qwen。"""
    catalog = json.loads(Path(__file__).with_name("yoloe_targets.json").read_text(encoding="utf-8"))
    key = " ".join(scene.split()).casefold()
    for entry in catalog["scenes"]:
        if key in (alias.casefold() for alias in entry["aliases"]):
            return tuple(entry["objects"])
    return None
