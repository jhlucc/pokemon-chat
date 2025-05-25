import json


# 从 JSON 文件中读取数据并提取人物名
def extract_person_names_from_json(input_file, output_file):
    # 读取 JSON 文件
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 获取 JSON 数据的所有顶级键作为人物名
    person_names = list(data.keys())

    # 将人物名写入输出文件，每行一个名字
    with open(output_file, "w", encoding="utf-8") as file:
        for name in person_names:
            file.write(name + "\n")

    print(f"人物名称已成功保存到 {output_file} 文件中。")


# 从 JSON 文件中提取指定属性的值并保存到 txt 文件中
def extract_property_values_to_txt(input_file, output_file, property_name):
    # 读取 JSON 文件
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 使用 set 来存储去重后的属性值
    values_set = set()

    # 提取指定属性值并添加到集合中
    for person_name, person_data in data.items():
        if property_name in person_data:
            value = person_data[property_name]
            if isinstance(value, list):
                for v in value:
                    values_set.add(v)  # 将列表中的每个元素加入集合
            else:
                values_set.add(value)  # 将单一值加入集合

    # 将去重后的属性值写入输出文件
    with open(output_file, "w", encoding="utf-8") as file:
        for value in values_set:
            file.write(f"{value}\n")

    print(f"{property_name} 属性值（去重）已保存到 {output_file} 文件中。")


# 调用函数，指定输入 JSON 文件和输出 TXT 文件的路径
# extract_person_names_from_json("region.json", "../en_data/Town.txt")

# 调用函数，指定 JSON 文件路径、输出文件路径和要提取的属性名
extract_property_values_to_txt("character.json", "1.txt", "region")
