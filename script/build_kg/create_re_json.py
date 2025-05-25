import json
import os


def append_relationship_type_to_json(input_file, output_file, start_entity_type, end_entity_type, rel_type, rel_name):
    # 如果 JSON 文件存在，读取现有数据
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        # 初始化一个空的 JSON 结构
        data = []

    # 检查是否已存在相同的 rel_type 类型
    existing_rel = next((item for item in data if item["rel_type"] == rel_type), None)

    if existing_rel:
        # 如果该类型已存在，则追加到现有的 rels 列表中
        with open(input_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    start_entity, end_entity, _ = parts
                    existing_rel["rels"].append({
                        "start_entity_name": start_entity,
                        "end_entity_name": end_entity
                    })
    else:
        # 如果该类型不存在，则创建新类型并添加到 data 中
        new_rel = {
            "start_entity_type": start_entity_type,
            "end_entity_type": end_entity_type,
            "rel_type": rel_type,
            "rel_name": rel_name,
            "rels": []
        }
        with open(input_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    start_entity, end_entity, _ = parts
                    if _ == rel_name:
                        new_rel["rels"].append({
                            "start_entity_name": start_entity,
                            "end_entity_name": end_entity
                        })
        data.append(new_rel)

    # 写回 JSON 文件
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"新的 {rel_name} 关系已成功追加到 {output_file} 文件中。")


append_relationship_type_to_json(
    "relations_data/has_pokemon.txt",
    "data/relations.json",
    start_entity_type="Person",
    end_entity_type="Pokémon",
    rel_type="has_pokemon",
    rel_name="拥有"
)
