import json
import os


# 处理不带属性的txt文本
def append_txt_to_json(input_file, output_file, label="Person"):
    entities = []

    # 读取文本文件中的实体
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            name = line.strip()
            if name:  # 跳过空行
                entities.append({"label": label, "name": name})

    # 如果 JSON 文件存在，先读取原有数据
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # 将新实体追加到原有数据中
    existing_data.extend(entities)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    print(f"{label} 实体已追加到 {output_file} 文件中。")


# append_txt_to_json("../en_data/identity.txt", "../data/entity.json", label="identity")


##############################################################################
# 把Pokémon实体内容追加到entity.json中
def convert_and_append_pokemon(input_file, entity_file):
    # 读取输入的宝可梦 JSON 数据
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    formatted_data = []

    # 遍历 JSON 数据，转换宝可梦信息
    for pokemon_name, pokemon_data in data.items():
        if "chinese_name" in pokemon_data:  # 判断是否为宝可梦实体
            formatted_data.append({
                "label": "Pokémon",
                "name": {
                    "chinese_name": pokemon_data.get("chinese_name", ""),
                    "japanese_name": pokemon_data.get("japanese_name", ""),
                    "english_name": pokemon_data.get("english_name", ""),
                    "ability": ", ".join(pokemon_data.get("ability", [])),
                    "hidden_ability": ", ".join(pokemon_data.get("隐藏特性", [])),
                    "height": pokemon_data.get("height", ""),
                    "weight": pokemon_data.get("weight", ""),
                    "evolution_level": pokemon_data.get("进化等级", ""),
                    "attr_ability": pokemon_data.get("属性相性", {})
                }
            })

    # 如果 entity.json 文件存在，先读取已有数据
    if os.path.exists(entity_file):
        with open(entity_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # 将新数据追加到已有数据中
    existing_data.extend(formatted_data)

    # 写回到 entity.json 文件
    with open(entity_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    print(f"宝可梦数据已成功追加到 {entity_file} 文件中。")


# convert_and_append_pokemon("pokemon_detail.json", "entity.json")


##############################################################################
# 把person 属性加到json中
def convert_and_append_person(input_file, entity_file):
    # 读取输入的 JSON 数据
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    formatted_data = []

    # 遍历 JSON 数据，转换人物信息
    for person_name, person_data in data.items():
        if "gender" in person_data:  # 判断是否为 Person 实体
            formatted_data.append({
                "label": "Person",
                "name": {
                    "chinese_name": person_data.get("chinese_name", ""),
                    "japanese_name": person_data.get("japanese_name", ""),
                    "english_name": person_data.get("english_name", ""),
                    "gender": person_data.get("gender", "")
                }
            })

    if os.path.exists(entity_file):
        with open(entity_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    existing_data.extend(formatted_data)
    with open(entity_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    print(f"人物数据已成功追加到 {entity_file} 文件中。")


append_txt_to_json("entity_data/Region.txt", "data/entities.json", label="Region")
