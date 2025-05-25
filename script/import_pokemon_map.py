import csv
import mysql.connector
from configs import  settings
# 数据库配置
db_config = {
    'host': 'localhost',
    'port': 3307,
    'user': 'root',
    'password': '123456',
    'database': 'langgraph',
    'charset': 'utf8mb4'
}

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 删除旧表（如果存在）
cursor.execute("DROP TABLE IF EXISTS pokemon_locations")

# 新建表，支持中文
cursor.execute("""
CREATE TABLE pokemon_locations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pokemon_region VARCHAR(255),
    real_location VARCHAR(255),
    pokemon_list TEXT,
    latitude DOUBLE,
    longitude DOUBLE,
    real_address VARCHAR(255),
    exact_match VARCHAR(10)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
""")

# 读取 CSV 并插入数据
with open(settings.MAP_DATA, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cursor.execute("""
            INSERT INTO pokemon_locations (
                pokemon_region, real_location, pokemon_list, latitude, longitude, real_address, exact_match
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            row["宝可梦地区"] or None,
            row["现实地区"] or None,
            row["pokemon"] or None,
            float(row["纬度"]) if row["纬度"] else None,
            float(row["经度"]) if row["经度"] else None,
            row["推荐地址"] or None,
            row["是否精确匹配"] or None
        ))

conn.commit()
cursor.close()
conn.close()
print("✅ CSV 已成功导入 MySQL！")
