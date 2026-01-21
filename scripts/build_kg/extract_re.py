# -*- coding: utf-8 -*-
import json


# 生成关系文本
# 读取txt文件，并提取指定关系写入新文件
def extract_relationships(input_file, output_file, target_relations):
    extracted_lines = []

    # 读取原始文件
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            # 检查行中是否包含目标关系
            for relation in target_relations:
                if relation in line:
                    extracted_lines.append(line.strip())  # 去除多余换行符，添加到列表中
                    break

    # 将提取的行写入新的文件
    with open(output_file, "w", encoding="utf-8") as file:
        for line in extracted_lines:
            file.write(line + "\n")

    print(f"已成功提取指定关系，并保存到 {output_file} 文件中。")


# extract_relationships("newrelation.txt", "../re_data/relative.txt", ["亲戚"])

##################################################################### 实体与实体########################

# 从 JSON 文件中提取有效的 chinese_name 和 region，生成 "name, region, 来自" 格式并保存到 txt 文件
def extract_come_from_relation(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    lines = []

    # 遍历每个实体，提取 chinese_name 和 region
    for entity_name, entity_data in data.items():
        chinese_name = entity_data.get("chinese_name", "")
        region = entity_data.get("进化", "")

        # 过滤条件：region 不能为 "Unknown" 或空值
        if chinese_name and region and region.lower() != "None" and chinese_name.lower() != "None":
            lines.append(f"{chinese_name},{region},进化")

    # 将结果写入到 txt 文件中
    with open(output_file, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")

    print(f"已成功提取 '来自' 关系并保存到 {output_file} 文件中。")


# extract_come_from_relation("pokemon_detail.json", "has_type.txt")

##################################################################### 实体与属性列表########################

def extract_pokemon_relation(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    lines = []

    # 遍历每个实体，提取 chinese_name 和 pokemon 列表
    for entity_name, entity_data in data.items():
        chinese_name = entity_data.get("chinese_name", "")
        pokemon_list = entity_data.get("character", [])

        # 如果 pokemon 列表不为空，则生成 "name, pokemon, 拥有" 格式的记录
        if chinese_name and pokemon_list:
            for pokemon in pokemon_list:
                lines.append(f"{chinese_name},{pokemon},有哪些名人")

    # 将结果写入到 txt 文件中
    with open(output_file, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")

    print(f"已成功提取 '拥有' 关系并保存到 {output_file} 文件中。")


# extract_pokemon_relation("../raw_data/region.json", "has_celebrity.txt")


def extract_person_pokemon(input_file, output_file):
    # 读取 JSON 文件
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 提取关系并保存到 TXT 文件
    with open(output_file, "w", encoding="utf-8") as output_file:
        for person, info in data.items():
            chinese_name = info["chinese_name"]
            pokemon_list = info["pokemon"]
            for pokemon in pokemon_list:
                # 写入关系：人物, 宝可梦, 拥有
                output_file.write(f"{chinese_name},{pokemon},拥有\n")


extract_person_pokemon("raw_data/character.json", "out.txt")
