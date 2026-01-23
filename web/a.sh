#!/bin/bash

SRC_DIR="./src"
API_PREFIXES=("/chat/" "/data/" "/graph/" "/agent/" "/tools/" "/model/" "/config/" "/api/")

echo "🔍 正在扫描 API 接口调用位置..."
echo "------------------------------------"

for prefix in "${API_PREFIXES[@]}"; do
  echo -e "\n📌 查找接口路径包含 '$prefix'"
  grep -rn --include \*.vue --include \*.js "$prefix" "$SRC_DIR" || echo "❌ 没找到 $prefix"
done

echo -e "\n✅ 扫描完成！"
