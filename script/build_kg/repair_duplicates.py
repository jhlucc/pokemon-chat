# 定义输入文件和输出文件路径
input_file = 'raw_data/relation.txt'  # 替换为你的文件路径
output_file = 'raw_data/final_relation.txt'  # 替换为输出文件路径

unique_lines = []
seen = set()
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)
with open(output_file, 'w', encoding='utf-8') as file:
    for line in unique_lines:
        file.write(line + '\n')
